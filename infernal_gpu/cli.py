"""infernal-gpu command line: one entry point for the package's commands.

    infernal-gpu calibrate IN.cm OUT.cm [--gpu N]             GPU replacement for cmcalibrate
    infernal-gpu search --out-dir D --cm-lib LIB --fasta F --infernal-bin DIR --n-models N [-- scanner options]
                                                             filter-free search, streamed native refinement
    infernal-gpu hybrid --cm LIB --fasta F --out-dir D --envdump-cmsearch X --infernal-bin DIR [--filter relaxed]
                                                             native HMM filters + GPU CM stages
    infernal-gpu tblout --run-dir D --out hits.tbl            export standard cmsearch tblout
    infernal-gpu refine --hits C --fasta F --cm-lib LIB --infernal-bin DIR --n-models N --out-dir D
                                                             native rescan of candidate regions
    infernal-gpu patch-infernal INFERNAL_SOURCE_DIR           add the opt-in envelope output (for hybrid)

The bundled GPU programs are used unless --scanner is given. Each command prints its own --help.
Native Infernal 1.1.5 (cmsearch, cmfetch) must be installed; search and hybrid take its bin directory.
"""
import importlib, os, sys

from . import __version__

COMMANDS = {'calibrate': 'calibrate', 'search': 'search', 'hybrid': 'hybrid', 'tblout': 'tblout',
            'refine': 'refine', 'patch-infernal': 'patch_infernal'}
# Measured production configuration of the GPU engine (callers' settings take precedence).
ENGINE_ENV = dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', GPU_TILE_POLICY='wave', GPU_PACKED_BEGL='auto',
                  GPU_PACK_ALL='auto', GPU_BLOCK_SIZE='128')
# Default scanner options for filter-free search (calibrated CYK filter, NULL3 gates, 0.25-bit margin).
SEARCH_DEFAULTS = ['--candidate-evalue', '1', '--cascade-cyk-p', '1e-4', '--null3-gate', '--cyk-margin', '0.25']


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == '--version':
        print('infernal-gpu', __version__)
        return 0
    if not argv or argv[0] in ('-h', '--help', 'help') or argv[0] not in COMMANDS:
        print(__doc__.strip())
        return 0 if not argv or argv[0] in ('-h', '--help', 'help') else 2
    cmd, rest = argv[0], argv[1:]
    for k, v in ENGINE_ENV.items():
        os.environ.setdefault(k, v)
    if cmd == 'search' and not any(a in ('-h', '--help') for a in rest):
        # scanner options follow '--'; add each default the caller did not set
        head, tail = (rest[:rest.index('--')], rest[rest.index('--') + 1:]) if '--' in rest else (rest, [])
        extra, i = [], 0
        while i < len(SEARCH_DEFAULTS):
            flag = SEARCH_DEFAULTS[i]
            takes_value = i + 1 < len(SEARCH_DEFAULTS) and not SEARCH_DEFAULTS[i + 1].startswith('--')
            if flag not in tail:
                extra += SEARCH_DEFAULTS[i:i + 1 + takes_value]
            i += 1 + takes_value
        rest = head + ['--'] + extra + tail
    sys.argv = ['infernal-gpu ' + cmd] + rest
    return importlib.import_module('.' + COMMANDS[cmd], __package__).main()


if __name__ == '__main__':
    sys.exit(main())
