"""Add an opt-in envelope dump to a fresh copy of Infernal 1.1.5.

The original baseline build is never touched. When INFERNAL_ENVDUMP_FILE is
set, cm_Pipeline writes every envelope that would enter the final CM stage
(after the configured HMM filters, and after F6 if F6 is on) and skips the
CM stages. Without the variable the code path is unchanged. Use --cpu 0:
the static FILE handle is not thread safe.
Columns: target, window_start, window_end, in_rc, env_start, env_end
(window-relative 1-based; after revcomp easel swaps start/end, so the
original coordinate of window position p is window_start - p + 1).

    infernal-gpu patch-infernal INFERNAL_SOURCE_DIR   (then ./configure && make in that directory)
"""
import argparse
from pathlib import Path

DUMP = r'''    { /* INFERNAL_GPU envelope dump (infernal-gpu hybrid front end) */
      static FILE *envdump_fp = NULL;
      static int   envdump_checked = 0;
      if(! envdump_checked) {
        char *envdump_path = getenv("INFERNAL_ENVDUMP_FILE");
        envdump_checked = 1;
        if(envdump_path != NULL) {
          envdump_fp = fopen(envdump_path, "w");
          if(envdump_fp == NULL) ESL_FAIL(eslFAIL, pli->errbuf, "cannot open INFERNAL_ENVDUMP_FILE");
        }
      }
      if(envdump_fp != NULL) {
        for(i = 0; i < nenv; i++)
          fprintf(envdump_fp, "%s\t%" PRId64 "\t%" PRId64 "\t%d\t%" PRId64 "\t%" PRId64 "\n",
                  sq2search->name, sq2search->start, sq2search->end, in_rc, es[i], ee[i]);
        fflush(envdump_fp);
        if(es != NULL) { free(es); es = NULL; }
        if(ee != NULL) { free(ee); ee = NULL; }
        nenv = 0;
        continue;
      }
    }
'''


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('infernal_source_dir', help='unpacked source of a separate copy of Infernal 1.1.5')
    a = p.parse_args()
    src = Path(a.infernal_source_dir) / 'src' / 'cm_pipeline.c'
    text = src.read_text()
    anchor = '    if(pli->do_time_F6) return status;\n'
    assert text.count(anchor) == 1
    assert 'INFERNAL_ENVDUMP_FILE' not in text
    src.write_text(text.replace(anchor, DUMP + anchor))
    print('patched', src)


if __name__ == '__main__':
    main()
