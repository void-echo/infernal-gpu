"""Rescan GPU candidate regions with Infernal to obtain individual corrected hits.

This is a CPU refinement stage, not GPU traceback. Candidates are selected
without consulting a reference hit list. Keep the original library-wide search
space using explicit -Z, even when only a diagnostic subset is processed.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_fasta(path):
    seqs = {}
    name = None
    with open(path) as f:
        for line in f:
            if line.startswith('>'):
                name = line[1:].split()[0]
                if name in seqs:
                    raise ValueError('Duplicate FASTA name: '+name)
                seqs[name] = []
            elif line.strip():
                seqs[name].append(line.strip())
    return {k: ''.join(v).upper() for k, v in seqs.items()}


def revcomp(s):
    return s.translate(str.maketrans('ACGTURYMKSWHBVDNX', 'TGCAAYRKMSWDVBHNX'))[::-1]


def merge_intervals(intervals):
    out = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1]+1:
            out[-1] = (out[-1][0], max(hi, out[-1][1]))
        else:
            out.append((lo, hi))
    return out


def map_coordinates(lo, hi, strand, start, end):
    return (lo+start-1, lo+end-1) if strand == '+' else (hi-start+1, hi-end+1)


def partition_regions(mapping, target_nt):
    """Partition complete, already-merged records without splitting any region.

    Stable input order is retained. A record larger than target_nt gets its
    own chunk, so target_nt is a target rather than a truncation limit.
    """
    if target_nt < 0:
        raise ValueError('region chunk target must be nonnegative')
    chunks, chunk, residues = [], {}, 0
    for name, interval in mapping.items():
        lo, hi = interval[2:]
        length = hi-lo+1
        if target_nt and chunk and residues+length > target_nt:
            chunks.append(chunk)
            chunk, residues = {}, 0
        chunk[name] = interval
        residues += length
    if chunk:
        chunks.append(chunk)
    return chunks


def run_region_tasks(tasks, jobs, worker):
    """One global pool for all families/chunks; workers never create pools."""
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(worker, tasks))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hits', required=True)
    p.add_argument('--fasta', required=True)
    p.add_argument('--cm-lib', required=True)
    p.add_argument('--infernal-bin', required=True)
    p.add_argument('--n-models', required=True, type=int)
    p.add_argument('--candidate-cutoff', type=float, default=1.)
    p.add_argument('--report-evalue', type=float, default=1.)
    p.add_argument('--families', help='Comma-separated diagnostic subset')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--cpu', type=int, default=1)
    p.add_argument('--region-chunk-nt', type=int, default=0,
                   help='Target residues per task after padding/merge; never splits a record. '
                        '0 keeps one task per model; chunking requires --cpu 1.')
    p.add_argument('--no-cyk-filter', action='store_true',
                   help='Re-search with cmsearch --noF6 (for a GPU scan without the CYK cascade, '
                        'i.e. Infernal --nohmm --noF6 semantics)')
    p.add_argument('--final-scoring', choices=['qdb', 'hmmband'], default='qdb',
                   help='qdb: native --nohmm final stage (default). hmmband: native HMM-banded final '
                        'stage with all HMM filters off and F5 P=1, used behind an HMM front end')
    p.add_argument('--pad', type=int, default=-1,
                   help='residues added on each side of a candidate before refinement; -1 (default) = model W. '
                        'Scanner candidates already start W-1 residues before their first gated endpoint, '
                        'so 0 suffices when every reported hit ends at a gated endpoint')
    p.add_argument('--out-dir', required=True)
    a = p.parse_args(argv)
    if a.n_models < 1 or a.jobs < 1 or a.cpu < 1:
        p.error('n-models, jobs and cpu must be positive')
    if a.final_scoring == 'hmmband' and a.no_cyk_filter:
        p.error('--no-cyk-filter applies to qdb final scoring; hmmband always skips F6')
    if a.region_chunk_nt < 0 or (a.region_chunk_nt and a.cpu != 1):
        p.error('region-chunk-nt must be nonnegative; chunked tasks require --cpu 1')
    root = Path(a.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    seqs = read_fasta(a.fasta)
    z_mb = 2 * sum(map(len, seqs.values())) * a.n_models / 1e6
    families = set(a.families.split(',')) if a.families else None
    candidates = defaultdict(list)
    with open(a.hits) as f:
        for row in csv.reader(f, delimiter='\t'):
            if not row or row[0].startswith('#'):
                continue
            target, cm, acc, frm, to, strand, score, ev = row[:8]
            acc = acc or '-'   # model without ACC line: fetch by name; tblout writes '-'
            if float(ev) > a.candidate_cutoff or (families and cm not in families):
                continue
            target = target.split()[0]
            lo, hi = sorted((int(frm), int(to)))
            if strand not in ('+', '-') or not 1 <= lo <= hi <= len(seqs[target]):
                raise ValueError('Invalid candidate coordinates: '+str(row))
            candidates[(cm, acc)].append((target, strand, lo, hi))
    manifest = dict(settings=vars(a), z_mb=z_mb, original_residues=sum(map(len,seqs.values())),
                    implementation_sha256=digest(__file__),
                    inputs={n:dict(path=str(Path(v).resolve()), sha256=digest(v))
                            for n,v in [('hits',a.hits),('fasta',a.fasta),('cm_lib',a.cm_lib)]})
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')

    def write_regions(path, mapping):
        with open(path, 'w') as f:
            for name, (target,strand,lo,hi) in mapping.items():
                sequence = seqs[target][lo-1:hi]
                if strand == '-':
                    sequence = revcomp(sequence)
                f.write('>'+name+'\n'+sequence+'\n')

    def prepare_model(item):
        ordinal, ((cm, acc), entries) = item
        d = root/f'cm_{ordinal:04d}'
        d.mkdir()
        cmfile = d/'model.cm'
        with open(cmfile, 'w') as f, open(d/'cmfetch.stderr', 'w') as err:
            subprocess.run([str(Path(a.infernal_bin)/'cmfetch'), a.cm_lib, acc if acc!='-' else cm],
                           stdout=f, stderr=err, check=True)
        # W comes from the CM header, never from reference coordinates.
        with open(cmfile) as f:
            w = next(int(l.split()[1]) for l in f if l.startswith('W '))
        grouped = defaultdict(list)
        for target, strand, lo, hi in entries:
            pad = w if a.pad < 0 else a.pad
            grouped[(target,strand)].append((max(1,lo-pad), min(len(seqs[target]),hi+pad)))
        mapping = {}
        for (target,strand), intervals in sorted(grouped.items()):
            for lo,hi in merge_intervals(intervals):
                prefix = f'cm_{ordinal:04d}_' if a.region_chunk_nt else ''
                name = f'{prefix}region_{len(mapping):06d}'
                mapping[name] = (target,strand,lo,hi)
        write_regions(d/'regions.fa', mapping)
        (d/'regions.json').write_text(json.dumps(mapping,indent=2)+'\n')
        return dict(ordinal=ordinal, cm=cm, acc=acc, directory=d, cmfile=cmfile,
                    W=w, entries=entries, mapping=mapping)

    def search_task(task):
        model, d, mapping = task['model'], task['directory'], task['mapping']
        fafile, tbl = d/'regions.fa', d/'refined.tbl'
        stage = (['--noF1', '--noF2', '--noF2b', '--noF3', '--noF3b', '--noF4', '--noF4b', '--F5', '1', '--noF6']
                 if a.final_scoring == 'hmmband' else ['--nohmm', *(['--noF6'] if a.no_cyk_filter else [])])
        command = [str(Path(a.infernal_bin)/'cmsearch'), *stage, '--nohmmonly',
                   '--toponly', '--notrunc', '--noali', '--cpu', str(a.cpu),
                   '--smxsize', '32768', '-Z', str(z_mb), '-E', str(a.report_evalue),
                   '--tblout', str(tbl), str(model['cmfile']), str(fafile)]
        t0 = time.monotonic()
        with open(d/'search.stdout','w') as out, open(d/'search.stderr','w') as err:
            subprocess.run(command, stdout=out, stderr=err, check=True)
        rows = []
        with open(tbl) as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                r = line.split()
                target,strand,lo,hi = mapping[r[0]]
                if r[9] != '+':
                    raise ValueError('Unexpected minus hit under --toponly')
                frm,to = map_coordinates(lo,hi,strand,int(r[7]),int(r[8]))
                rows.append([target,model['cm'],model['acc'],frm,to,strand,float(r[14]),float(r[15])])
        t1 = time.monotonic()
        record = dict(chunk=task['chunk'], regions=len(mapping), region_ids=list(mapping),
                      residues=sum(hi-lo+1 for _,_,lo,hi in mapping.values()),
                      cpu_search_wall_s=t1-t0, started_s=t0-start, finished_s=t1-start,
                      reported_hits=len(rows), command=command)
        if a.region_chunk_nt:
            (d/'result.json').write_text(json.dumps(record,indent=2)+'\n')
        return task,rows,record

    def finish_model(model, outputs):
        outputs.sort(key=lambda result: result[0]['chunk'] or 0)
        rows = [row for _,task_rows,_ in outputs for row in task_rows]
        chunks = [record for _,_,record in outputs]
        record = dict(cm=model['cm'], accession=model['acc'], candidates=len(model['entries']),
                      regions=len(model['mapping']), W=model['W'],
                      residues=sum(hi-lo+1 for _,_,lo,hi in model['mapping'].values()),
                      cpu_search_wall_s=max(r['finished_s'] for r in chunks)-min(r['started_s'] for r in chunks),
                      reported_hits=len(rows))
        if a.region_chunk_nt:
            record.update(chunks=chunks, cpu_search_wall_sum_s=sum(r['cpu_search_wall_s'] for r in chunks))
            # Keep a per-model tblout for existing readers; exact commands and
            # individual Infernal outputs remain in the chunk directories.
            with open(model['directory']/'refined.tbl','w') as combined:
                combined.write('# Combined chunk tblout; commands in chunk result.json files.\n')
                for task,_,_ in outputs:
                    with open(task['directory']/'refined.tbl') as f:
                        for line in f:
                            if not line.startswith('#') and line.strip():
                                combined.write(line)
        else:
            record['command'] = chunks[0]['command']
        (model['directory']/'result.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({k:v for k,v in record.items() if k not in ('command','chunks')}),flush=True)
        return rows,record

    items = list(enumerate(sorted(candidates.items())))
    if a.region_chunk_nt:
        models = [prepare_model(item) for item in items]
        tasks = []
        for model in models:
            for chunk, mapping in enumerate(partition_regions(model['mapping'], a.region_chunk_nt)):
                d = model['directory']/f'chunk_{chunk:04d}'
                d.mkdir()
                write_regions(d/'regions.fa', mapping)
                (d/'regions.json').write_text(json.dumps(mapping,indent=2)+'\n')
                tasks.append(dict(model=model, chunk=chunk, directory=d, mapping=mapping))
        # Start larger estimated jobs first, across families as well as chunks.
        # W is already available and provides a simple CM-work proxy.
        tasks.sort(key=lambda t: sum(hi-lo+1 for _,_,lo,hi in t['mapping'].values())*t['model']['W'],
                   reverse=True)
        grouped_outputs = defaultdict(list)
        for result in run_region_tasks(tasks, a.jobs, search_task):
            grouped_outputs[result[0]['model']['ordinal']].append(result)
        completed = [finish_model(model, grouped_outputs[model['ordinal']]) for model in models]
        manifest.update(search_tasks=len(tasks), scheduler='global_pool_descending_residues_times_W',
                        search_workers_per_task=1)
    else:
        def process_model(item):
            model = prepare_model(item)
            task = dict(model=model, chunk=None, directory=model['directory'], mapping=model['mapping'])
            return finish_model(model, [search_task(task)])
        completed = run_region_tasks(items, a.jobs, process_model)

    records, hits = [], []
    for rows, record in completed:
        hits.extend(rows)
        records.append(record)
    with open(root/'refined_hits.tsv','w') as f:
        f.write('#target_name\tcm_name\tcm_acc\tseq_from\tseq_to\tstrand\tscore\tevalue\n')
        csv.writer(f,delimiter='\t').writerows(sorted(hits,key=lambda r:r[7]))
    manifest.update(refinement_wall_s=time.monotonic()-start, models=records,
                    requested_families_without_candidates=sorted((families or set())-{cm for cm,_ in candidates}),
                    reported_hits=len(hits), significant_hits=sum(r[7]<.01 for r in hits))
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['refinement_wall_s','reported_hits','significant_hits']}))


if __name__ == '__main__':
    main()
