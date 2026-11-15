"""Sensitive search with Infernal's HMM filters on the CPU and the CM stages on the GPU.

    infernal-gpu hybrid --cm models.cm --fasta targets.fa --out-dir run \
        --filter relaxed --envdump-cmsearch ENVDUMP/src/cmsearch --infernal-bin INFERNAL/src \
        [--gpu 0] [--jobs 16]

1. Native HMM front end (per model, CPU): the envelope-dump build of cmsearch
   (infernal-gpu patch-infernal applied to a separate copy of Infernal 1.1.5) runs the
   requested HMM filter stages with --noF6 and writes the envelopes that would enter the CM stages.
2. GPU: envelopes are merged per model/target/strand, oriented, and every model scans only its own
   segments (--model-prefix) with QDB Inside; E-values use the original search space (cmsearch
   semantics: per-model, Z from the target database).
3. Native refinement (CPU) of the GPU candidates with QDB final scoring, and standard tblout.
The models must be calibrated (cmcalibrate or infernal-gpu calibrate). --filter is one of
default, mid, relaxed (--mid --Fmid 0.1), or --filter-args gives explicit cmsearch filter options.
Python 3.8+, standard library.
"""
import argparse, json, os, resource, subprocess, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BIN = Path(__file__).resolve().parent / 'bin'
FILTERS = {'default': [], 'mid': ['--mid'], 'relaxed': ['--mid', '--Fmid', '0.1']}


def read_fasta(path):
    seqs, name = {}, None
    for line in open(path):
        if line.startswith('>'):
            name = line[1:].split()[0]; seqs[name] = []
        elif line.strip():
            seqs[name].append(line.strip())
    return {k: ''.join(v).upper() for k, v in seqs.items()}


def revcomp(s):
    return s.translate(str.maketrans('ACGTURYMKSWHBVDNX', 'TGCAAYRKMSWDVBHNX'))[::-1]


def split_library(path):
    models, cur, name = [], [], None
    for line in open(path):
        if line.startswith('INFERNAL1') and cur:
            models.append((name, ''.join(cur))); cur, name = [], None
        if name is None and line.startswith('NAME'):
            name = line.split()[1]
        cur.append(line)
    if cur: models.append((name, ''.join(cur)))
    return models


def merged_envelopes(path, L):
    """Envelope-dump rows -> merged (target, strand, lo, hi) in original coordinates."""
    env = defaultdict(list)
    for line in open(path):
        t, ws, we, rc, es, ee = line.split()
        ws, es, ee = int(ws), int(es), int(ee)
        lo, hi, st = (ws + es - 1, ws + ee - 1, '+') if rc == '0' else (ws - ee + 1, ws - es + 1, '-')
        env[(t, st)].append((max(1, lo), min(L[t], hi)))
    out = []
    for (t, st), iv in sorted(env.items()):
        iv.sort(); m = [list(iv[0])]
        for lo, hi in iv[1:]:
            if lo <= m[-1][1] + 1: m[-1][1] = max(m[-1][1], hi)
            else: m.append([lo, hi])
        out += [(t, st, lo, hi) for lo, hi in m]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cm', required=True, type=Path, help='calibrated CM file (one model or a library)')
    p.add_argument('--fasta', required=True, type=Path)
    p.add_argument('--out-dir', required=True, type=Path)
    p.add_argument('--filter', choices=FILTERS, default='relaxed')
    p.add_argument('--filter-args', help='explicit cmsearch filter options (overrides --filter)')
    p.add_argument('--envdump-cmsearch', required=True)
    p.add_argument('--infernal-bin', required=True, help='directory with the unmodified cmsearch/cmfetch')
    p.add_argument('--scanner', default=str(BIN / 'infernal-gpu-scan'), help='GPU scanner (default: the bundled one)')
    p.add_argument('--gpu', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    p.add_argument('--jobs', type=int, default=16)
    p.add_argument('--report-evalue', type=float, default=10.0)
    a = p.parse_args()
    out = a.out_dir; out.mkdir(parents=True, exist_ok=False)
    flags = a.filter_args.split() if a.filter_args else FILTERS[a.filter]
    seqs = read_fasta(a.fasta)
    L = {k: len(v) for k, v in seqs.items()}
    total = sum(L.values())
    models = split_library(a.cm)
    (out / 'models').mkdir()
    for n, text in models:
        (out / 'models' / f'{n}.cm').write_text(text)
    timing = {}

    # 1. native HMM front end, one model per task
    def front(model):
        n = model[0]
        env = dict(os.environ, INFERNAL_ENVDUMP_FILE=str(out / 'models' / f'{n}.env'))
        cmd = [a.envdump_cmsearch, *flags, '--noF6', '--notrunc', '--nohmmonly', '--noali', '--cpu', '0',
               '-E', str(a.report_evalue), str(out / 'models' / f'{n}.cm'), str(a.fasta)]
        with open(out / 'models' / f'{n}.front.out', 'w') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, env=env)
    t0 = time.monotonic(); r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(front, models))
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    timing['front_wall_s'] = time.monotonic() - t0
    timing['front_cpu_core_s'] = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)

    # 2. GPU scan of the envelope segments
    names, seg_res = {}, 0
    with open(out / 'scan_models.cm', 'w') as cmf, open(out / 'segments.fa', 'w') as fa:
        for n, text in models:
            segs = merged_envelopes(out / 'models' / f'{n}.env', L)
            if not segs: continue
            cmf.write(text)
            for i, (t, st, lo, hi) in enumerate(segs):
                s = seqs[t][lo - 1:hi]
                fa.write(f'>{n}|s{i}\n{s if st == "+" else revcomp(s)}\n')
                names[f'{n}|s{i}'] = (t, st, lo, hi)
                seg_res += hi - lo + 1
    timing['envelope_residues'] = seg_res
    cand = []
    t0 = time.monotonic()
    if names:
        env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(a.gpu), GPU_BLOCK_SIZE='128',
                   GPU_TILE_POLICY='wave', GPU_PACKED_BEGL='auto', GPU_PACK_ALL='auto')
        cmd = [a.scanner, str(out / 'scan_models.cm'), str(out / 'segments.fa'), '--toponly', '--model-prefix',
               '--evalue-residues', str(total), '--evalue-models', '1', '--candidate-evalue', str(a.report_evalue),
               '--null3-gate', '--cyk-margin', '0.25', '--hits-out', str(out / 'gpu_candidates.tsv')]
        with open(out / 'scan.stdout', 'w') as o, open(out / 'scan.stderr', 'w') as e:
            subprocess.run(cmd, env=env, stdout=o, stderr=e, check=True)
        for line in open(out / 'gpu_candidates.tsv'):
            if line.startswith('#') or not line.strip(): continue
            r = line.rstrip('\n').split('\t')
            t, st, lo, hi = names[r[0].split()[0]]
            i, j = int(r[3]), int(r[4])
            fr, to = (lo + i - 1, lo + j - 1) if st == '+' else (hi - i + 1, hi - j + 1)
            cand.append([t, r[1], r[2], fr, to, st, r[6], r[7]])
    timing['gpu_scan_wall_s'] = time.monotonic() - t0
    with open(out / 'candidates.tsv', 'w') as fo:
        fo.write('#target_name\tcm_name\tcm_acc\tseq_from\tseq_to\tstrand\tscore\tevalue\n')
        for r in cand: fo.write('\t'.join(map(str, r)) + '\n')

    # 3. native QDB refinement of the candidates (keeps W padding; envelope-clipped candidates
    #    must be allowed to extend), then standard tblout
    lib = out / 'refine_lib.cm'
    lib.write_text(''.join(t for _, t in models))
    subprocess.run([str(Path(a.infernal_bin) / 'cmfetch'), '--index', str(lib)], check=True, stdout=subprocess.DEVNULL)
    t0 = time.monotonic(); r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cmd = [sys.executable, '-m', 'infernal_gpu.refine', '--hits', str(out / 'candidates.tsv'),
           '--fasta', str(a.fasta), '--cm-lib', str(lib), '--infernal-bin', a.infernal_bin, '--n-models', '1',
           '--report-evalue', str(a.report_evalue), '--jobs', str(a.jobs), '--cpu', '1', '--region-chunk-nt', '1',
           '--candidate-cutoff', 'inf', '--final-scoring', 'qdb', '--no-cyk-filter', '--out-dir', str(out / 'refined')]
    with open(out / 'refine.stdout', 'w') as o, open(out / 'refine.stderr', 'w') as e:
        subprocess.run(cmd, stdout=o, stderr=e, check=True)
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    timing['refine_wall_s'] = time.monotonic() - t0
    timing['refine_cpu_core_s'] = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    subprocess.run([sys.executable, '-m', 'infernal_gpu.tblout', '--run-dir', str(out / 'refined'),
                    '--out', str(out / 'hits.tbl')], check=True)
    summary = dict(models=len(models), filter=flags, candidates=len(cand), **timing)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary), file=sys.stderr)


if __name__ == '__main__':
    main()
