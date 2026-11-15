"""End-to-end test of an installed infernal-gpu (NVIDIA GPU and Infernal 1.1.5 required).

    INFERNAL_BIN=/path/to/infernal-1.1.5/src python tests/test_infernal_gpu.py    (or: pytest tests)

1. calibrate: both models of data/models.cm are recalibrated on the GPU from a copy without ECM
   lines; in every mode, the score at which E = 0.01 in a 100-Mb search must agree with
   cmcalibrate's calibration (the shipped file) within 0.2 bits (measured: 0.002-0.024 bits, and
   0.098 for 5S_rRNA local Inside; a default cmcalibrate run's own sampling error is about 1 bit).
2. search: filter-free search of both models against data/mrum-slice.fa must report exactly the
   hits of cmsearch --nohmm --notrunc with the same search space (coordinates, scores, E-values).

Data: tRNA5.c.cm and 5S_rRNA.c.cm (calibrated by cmcalibrate) and three segments of mrum-genome.fa
(Methanobrevibacter ruminantium, NC_013790.1; 138 kb) from the Infernal 1.1.5 tutorial (BSD license).
"""
import math, os, subprocess, sys, tempfile
from pathlib import Path

DATA = Path(__file__).resolve().parent / 'data'
MODELS, FASTA = DATA / 'models.cm', DATA / 'mrum-slice.fa'
IGPU = [sys.executable, '-m', 'infernal_gpu.cli']   # the infernal-gpu installed for this Python


def run(cmd, cwd):
    # cwd is a scratch directory, so a source checkout cannot shadow the installed package
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def thresholds(cm_file):
    """{(model, mode): bit score at which E = 0.01 in a 100-Mb search}"""
    out, name = {}, None
    for line in open(cm_file):
        f = line.split()
        if line.startswith('INFERNAL1'): name = None
        elif f and f[0] == 'NAME' and name is None: name = f[1]
        elif f and f[0].startswith('ECM'):
            lam, mu, dbsize, nrand = float(f[1]), float(f[2]), float(f[4]), float(f[5])
            out[(name, f[0])] = mu + math.log(nrand * 1e8 / dbsize / 0.01) / lam
    return out


def hits(tblout):
    """(target, model, from, to, strand, score, E-value) of every tblout row"""
    rows = [l.split() for l in open(tblout) if l.strip() and not l.startswith('#')]
    return sorted((r[0], r[2], int(r[7]), int(r[8]), r[9], float(r[14]), float(r[15])) for r in rows)


def test_calibrate():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / 'uncal.cm').write_text(''.join(l for l in open(MODELS) if not l.startswith('ECM')))
        run(IGPU + ['calibrate', td / 'uncal.cm', td / 'gpu.cm'], td)
        ref, gpu = thresholds(MODELS), thresholds(td / 'gpu.cm')
        assert len(ref) == 8 and set(gpu) == set(ref)
        diff = max(abs(gpu[k] - ref[k]) for k in ref)
        print(f'calibrate: max |delta x(E=0.01)| over 2 models x 4 modes = {diff:.4f} bits')
        assert diff < 0.2


def test_search():
    bin_dir = os.environ.get('INFERNAL_BIN')
    assert bin_dir, 'set INFERNAL_BIN to the directory with Infernal 1.1.5 cmsearch and cmfetch'
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        run(IGPU + ['search', '--cm-lib', MODELS, '--n-models', 2, '--fasta', FASTA,
                    '--infernal-bin', bin_dir, '--parallel', 2, '--jobs', 2, '--out-dir', td / 'run'], td)
        residues = sum(len(l.strip()) for l in open(FASTA) if not l.startswith('>'))
        z = 2 * residues * 2 / 1e6   # both strands, two models (as infernal-gpu refine sets -Z)
        run([Path(bin_dir) / 'cmsearch', '--nohmm', '--nohmmonly', '--notrunc', '--noali', '--cpu', 2,
             '--smxsize', 32768, '-Z', z, '-E', 1, '--tblout', td / 'ref.tbl', '-o', os.devnull, MODELS, FASTA], td)
        got, ref = hits(td / 'run' / 'hits.tbl'), hits(td / 'ref.tbl')
        print(f'search: {len(got)} hits ({sum(h[-1] < 0.01 for h in got)} with E < 0.01, '
              f'{sum(h[4] == "-" for h in got)} on the minus strand); cmsearch --nohmm: {len(ref)}')
        assert ref and got == ref


if __name__ == '__main__':
    test_calibrate()
    test_search()
    print('all tests passed')
