"""GPU replacement for `cmcalibrate` (default protocol): calibrate CM E-values and write a model file.

    infernal-gpu calibrate IN.cm OUT.cm [--gpu 0]

IN.cm is a model or library from cmbuild (with or without ECM lines). The scanner is the bundled
infernal-gpu-calibrate-scan (built with -DCALIB_RECORDS). Every model is calibrated in cmcalibrate's
four modes on cmcalibrate's own default random sequences (data/cmcalibrate_default_seed181.fa in
this package: 160 x 10 kb from Infernal's genomic HMM, seed 181), and the four ECM lines are written
into OUT.cm in Infernal's format; everything else in the file is copied unchanged. OUT.cm can be used
directly by cmsearch/cmscan. --seqs selects other random sequences (e.g. longer ones for a more
precise calibration; dbsize is taken from their total length). Python 3.8+, standard library.
"""
import argparse, os, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODES = ('ECMLC', 'ECMGC', 'ECMLI', 'ECMGI')   # cm_file.c write order


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cm_in'); ap.add_argument('cm_out')
    ap.add_argument('--scanner', default=str(HERE / 'bin' / 'infernal-gpu-calibrate-scan'),
                    help='GPU scanner built with -DCALIB_RECORDS (default: the bundled one)')
    ap.add_argument('--gpu', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    ap.add_argument('--seqs', default=str(HERE / 'data' / 'cmcalibrate_default_seed181.fa'))
    ap.add_argument('--keep-tsv', help='also keep the per-mode calibration table here')
    ap.add_argument('--gtailn', type=int, default=250, help='hits/Mb fitted in glocal modes (cmcalibrate --gtailn)')
    ap.add_argument('--ltailn', type=int, default=750, help='hits/Mb fitted in local modes (cmcalibrate --ltailn)')
    a = ap.parse_args()

    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(a.gpu),
               GPU_TILE_POLICY='wave', GPU_PACKED_BEGL='auto', GPU_PACK_ALL='auto')
    with tempfile.TemporaryDirectory() as td:
        tsv = a.keep_tsv or str(Path(td) / 'calib.tsv')
        t0 = time.time()
        r = subprocess.run([a.scanner, a.cm_in, a.seqs, '--calibrate', tsv,
                            '--gtailn', str(a.gtailn), '--ltailn', str(a.ltailn)], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        wall = time.time() - t0
        if r.returncode != 0:
            sys.exit('calibration failed:\n' + '\n'.join(l for l in r.stderr.splitlines() if 'FATAL' in l or 'ERR' in l))
        calib = {}
        for line in open(tsv):
            if line.startswith('#'): continue
            f = line.rstrip('\n').split('\t')
            # cm_name cm_acc mode lambda mu_extrap mu_orig dbsize nrandhits tailp ...
            calib.setdefault(f[0], {})[f[2]] = (float(f[3]), float(f[4]), float(f[5]), int(float(f[6]) + 0.5), int(f[7]), float(f[8]))

    out, in_header, name, n = [], False, None, 0
    for line in open(a.cm_in):
        if line.startswith('INFERNAL1'):
            in_header, name = True, None
        if in_header:
            if line.startswith('NAME') and name is None: name = line.split()[1]
            if line.startswith('ECM'): continue
            if line.rstrip('\n') == 'CM':
                c = calib.get(name, {})
                if len(c) != 4: sys.exit(f'{name}: calibration incomplete')
                out += [f'{m}    {c[m][0]:.5f}  {c[m][1]:10.5f}  {c[m][2]:10.5f}  {c[m][3]:10d}  {c[m][4]:10d}  {c[m][5]:.6f}\n'
                        for m in MODES]
                in_header = False; n += 1
        out.append(line)
    Path(a.cm_out).write_text(''.join(out))
    print(f'{n} model(s) calibrated in {wall:.1f} s -> {a.cm_out}', file=sys.stderr)


if __name__ == '__main__':
    main()
