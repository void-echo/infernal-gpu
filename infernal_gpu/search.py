"""Filter-free search: run the GPU scanner and CPU candidate refinement concurrently.

The scanner is started with --hits-stream DIR (one candidate file per model,
published by rename as soon as that model is scanned, then DIR/DONE). Each
new file is refined at once by native cmsearch (infernal-gpu refine) while the GPU
keeps scanning later models, so refinement overlaps the scan instead of
following it. At the end the per-model outputs are merged (sorted by E-value),
the streamed candidates are checked to equal the scanner's --hits-out set, and
hits.tbl is written in cmsearch tblout format.

Example:
  infernal-gpu search --out-dir run1 --cm-lib Rfam.cm --fasta genome.fa \
    --infernal-bin infernal-1.1.5/src --n-models 4227 --parallel 4 --jobs 4
Arguments after '--' are passed to the scanner; infernal-gpu adds the defaults
--candidate-evalue 1 --cascade-cyk-p 1e-4 --null3-gate --cyk-margin 0.25 unless given."""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .tblout import export_tblout

BIN = Path(__file__).resolve().parent / 'bin'


def rows(path):
    return [l.rstrip('\n') for l in open(path) if l.strip() and not l.startswith('#')]


def main():
    argv = sys.argv[1:]
    scan_extra = []
    if '--' in argv:
        k = argv.index('--'); scan_extra = argv[k + 1:]; argv = argv[:k]
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--out-dir', required=True)
    p.add_argument('--scanner', default=str(BIN / 'infernal-gpu-scan'), help='GPU scanner (default: the bundled one)')
    p.add_argument('--cm-lib', required=True, help='CM library for refinement (cmfetch-indexed)')
    p.add_argument('--scan-cm', help='CM file the scanner searches (default: --cm-lib)')
    p.add_argument('--fasta', required=True)
    p.add_argument('--infernal-bin', required=True)
    p.add_argument('--n-models', type=int, required=True, help='E-value multiplicity for refinement')
    p.add_argument('--report-evalue', default='1')
    p.add_argument('--parallel', type=int, default=4, help='concurrent per-model refinements')
    p.add_argument('--jobs', type=int, default=4, help='cmsearch tasks inside each refinement')
    p.add_argument('--poll-s', type=float, default=2.0)
    p.add_argument('--no-cyk-filter', action='store_true', help='refine with cmsearch --noF6')
    a = p.parse_args(argv)

    out = Path(a.out_dir); out.mkdir(exist_ok=False)
    stream = out / 'stream'; stream.mkdir()
    parts = out / 'refined_parts'; parts.mkdir()
    t0 = time.monotonic()
    log = {'started': {}, 'finished': {}}

    def refine(f):
        d = parts / f.stem
        log['started'][f.stem] = time.monotonic() - t0
        cmd = [sys.executable, '-m', 'infernal_gpu.refine', '--hits', str(f), '--fasta', a.fasta,
               '--cm-lib', a.cm_lib, '--infernal-bin', a.infernal_bin, '--n-models', str(a.n_models),
               '--report-evalue', a.report_evalue, '--jobs', str(a.jobs), '--cpu', '1',
               '--region-chunk-nt', '1', '--candidate-cutoff', 'inf', '--out-dir', str(d),
               *(['--no-cyk-filter'] if a.no_cyk_filter else [])]
        with open(str(d) + '.log', 'w') as o:
            subprocess.run(cmd, stdout=o, stderr=subprocess.STDOUT, check=True)
        log['finished'][f.stem] = time.monotonic() - t0

    scan_cmd = [a.scanner, a.scan_cm or a.cm_lib, a.fasta, '--hits-out', str(out / 'candidates.tsv'),
                '--hits-stream', str(stream), *scan_extra]
    so, se = open(out / 'scan.stdout', 'w'), open(out / 'scan.stderr', 'w')
    scanner = subprocess.Popen(scan_cmd, stdout=so, stderr=se)
    seen, futures = set(), []
    with ThreadPoolExecutor(a.parallel) as ex:
        while True:
            for f in sorted(stream.glob('*.tsv')):
                if f.name not in seen:
                    seen.add(f.name); futures.append(ex.submit(refine, f))
            if scanner.poll() is not None and (stream / 'DONE').exists():
                for f in sorted(stream.glob('*.tsv')):
                    if f.name not in seen:
                        seen.add(f.name); futures.append(ex.submit(refine, f))
                break
            if scanner.poll() not in (None, 0):
                break
            time.sleep(a.poll_s)
        scan_s = time.monotonic() - t0
        for fu in futures: fu.result()
    so.close(); se.close()
    if scanner.returncode != 0:
        sys.exit(f'scanner exited {scanner.returncode}; see {out}/scan.stderr')

    streamed = sorted(r for f in stream.glob('*.tsv') for r in rows(f))
    full = sorted(rows(out / 'candidates.tsv'))
    if streamed != full:
        sys.exit(f'streamed candidates ({len(streamed)}) differ from --hits-out ({len(full)})')
    merged = []
    for f in sorted(parts.glob('*/refined_hits.tsv')):
        merged += [r.split('\t') for r in rows(f)]
    merged.sort(key=lambda r: float(r[7]))
    with open(out / 'refined_hits.tsv', 'w') as o:
        o.write('#target_name\tcm_name\tcm_acc\tseq_from\tseq_to\tstrand\tscore\tevalue\n')
        for r in merged: o.write('\t'.join(r) + '\n')
    summary = dict(wall_s=time.monotonic() - t0, scan_process_s=scan_s, candidates=len(full),
                   model_files=len(seen), refined_hits=len(merged),
                   refinement_tail_s=(time.monotonic() - t0) - scan_s, timeline=log, scan_command=scan_cmd)
    export_tblout(out, out / 'hits.tbl', require_complete=False)
    summary['wall_s'] = time.monotonic() - t0
    summary['refinement_tail_s'] = summary['wall_s'] - scan_s
    summary['tblout'] = str(out / 'hits.tbl')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('timeline', 'scan_command')}), flush=True)


if __name__ == '__main__':
    main()
