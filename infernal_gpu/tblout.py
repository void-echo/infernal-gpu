"""Export completed GPU+refinement results in Infernal cmsearch tblout format.

Preserves native model coordinates, bias, score, E-value and description.
Only the region name, target coordinates and strand are mapped back to the
original FASTA. The exported hit multiset must match refined_hits.tsv.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from .refine import map_coordinates


def export_tblout(run_dir, destination, require_complete=True):
    root, destination = Path(run_dir), Path(destination)
    streamed = (root / 'refined_parts').is_dir()
    marker = root / ('summary.json' if streamed else 'manifest.json')
    if require_complete and not marker.exists():
        raise ValueError('Expected a completed stream or refinement run')
    glob = 'refined_parts/*/cm_*/regions.json' if streamed else 'cm_*/regions.json'
    rows = []
    for mapping_file in sorted(root.glob(glob)):
        mapping = json.loads(mapping_file.read_text())
        # For chunked runs this is the per-model combined native table;
        # reading chunk tables as well would double-count all hits.
        table = mapping_file.parent / 'refined.tbl'
        for line in table.read_text().splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            fields = line.split(maxsplit=17)
            if len(fields) < 17:
                raise ValueError('Invalid native tblout row in '+str(table))
            if len(fields) == 17:
                fields.append('-')
            target, strand, lo, hi = mapping[fields[0]]
            start, end = int(fields[7]), int(fields[8])
            if fields[9] != '+' or not 1 <= start <= end <= hi-lo+1:
                raise ValueError('Invalid forward-oriented region hit in '+str(table))
            start, end = map_coordinates(lo, hi, strand, start, end)
            fields[0], fields[7], fields[8], fields[9] = target, str(start), str(end), strand
            rows.append(fields)
    observed = Counter((r[0], r[2], r[3], int(r[7]), int(r[8]), r[9], float(r[14]), float(r[15])) for r in rows)
    expected = Counter()
    for line in (root / 'refined_hits.tsv').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        r = line.split('\t')
        expected[(r[0], r[1], r[2], int(r[3]), int(r[4]), r[5], float(r[6]), float(r[7]))] += 1
    if observed != expected:
        raise ValueError('Native tables do not reproduce refined_hits.tsv; export aborted')
    rows.sort(key=lambda r: (float(r[15]), r[2], r[0], int(r[7]), int(r[8])))
    text = '# Infernal-GPU: native cmsearch refinement mapped to original sequence coordinates\n'
    text += '# target name accession query name accession mdl mdl from mdl to seq from seq to strand trunc pass gc bias score E-value inc description of target\n'
    text += '# Scores and E-values are from native Infernal; see run manifests for mode and search space.\n'
    text += ''.join(' '.join(r)+'\n' for r in rows)
    text += '# [ok]\n'
    destination.write_text(text)
    return len(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    print('Exported', export_tblout(a.run_dir, a.out), 'hits to', a.out)


if __name__ == '__main__':
    main()
