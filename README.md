# Infernal-GPU

Infernal-GPU is a GPU implementation of the covariance-model (CM) dynamic programming of
[Infernal](http://eddylab.org/infernal/) 1.1.5. It provides a replacement for `cmcalibrate` that
follows the same protocol and writes the same model file, and a CM search in which the GPU scores
every model against every sequence while Infernal's own `cmsearch` computes the final hits, so
scores, E-values and coordinates are Infernal's.

## Requirements

- Linux with an NVIDIA GPU and the CUDA toolkit (`nvcc`)
- CMake ≥ 3.24 (fetched by pip if missing) and Python ≥ 3.8
- Infernal 1.1.5 (`cmsearch`, `cmfetch`) for `search` and `hybrid`

## Install

    pip install infernal-gpu                                   # from PyPI; compiles the GPU programs
    pip install git+https://github.com/void-echo/infernal-gpu  # or from GitHub (or `pip install .` in a clone)

The GPU programs are compiled for the GPU of the build machine. To build for other or several
architectures (A100 = 80, RTX 30xx = 86, RTX 40xx = 89 with CUDA ≥ 11.8), name them; if CMake
does not find `nvcc`, also set `CUDACXX=/usr/local/cuda/bin/nvcc`:

    CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=80;86" pip install .

## Usage

Calibrate E-values of a model or library from `cmbuild`:

    infernal-gpu calibrate model.cm model.calibrated.cm --gpu 0

Calibrations on more random sequence are more accurate (sampling error falls with the square root of its
length). To calibrate on 16 Mb, ten times the default, compile cmcalibrate's sequence generator once
against an Infernal 1.1.5 source build and pass its output with `--seqs`:

    I=infernal-1.1.5
    cc -O2 -I$I/src -I$I/hmmer/src -I$I/easel tools/gen_calib_seqs.c $I/src/libinfernal.a \
       $I/hmmer/src/libhmmer.a $I/easel/libeasel.a -lm -lpthread -o gen_calib_seqs
    ./gen_calib_seqs 181 1600 10000 > random16Mb.fa          # seed, chunks, chunk length
    infernal-gpu calibrate model.cm model.calibrated.cm --seqs random16Mb.fa --gpu 0

Filter-free search of calibrated models (the hits of `cmsearch --nohmm`). E-values cover the
`--n-models` models of the library, as in `cmscan`; the GPU is chosen with `CUDA_VISIBLE_DEVICES`:

    infernal-gpu search --cm-lib Rfam.cm --n-models 4227 --fasta genome.fa \
        --infernal-bin infernal-1.1.5/src --out-dir run1

Sensitive search behind Infernal's HMM filters, with the CM stages on the GPU. It needs a second
copy of Infernal 1.1.5, patched once so that it writes out the envelopes that pass the filters:

    curl -LO http://eddylab.org/infernal/infernal-1.1.5.tar.gz
    tar xf infernal-1.1.5.tar.gz && mv infernal-1.1.5 infernal-envdump
    infernal-gpu patch-infernal infernal-envdump
    (cd infernal-envdump && ./configure && make)
    infernal-gpu hybrid --cm models.cm --fasta targets.fa --filter relaxed --gpu 0 --out-dir run2 \
        --envdump-cmsearch infernal-envdump/src/cmsearch --infernal-bin infernal-1.1.5/src

`--filter` is `default`, `mid` or `relaxed` (`--mid --Fmid 0.1`). Every command lists its options
with `--help`.

## Outputs

`calibrate` writes the model file with new `ECM` lines and everything else unchanged; it works with
`cmsearch` and `cmscan`. `search` and `hybrid` write `hits.tbl` in `cmsearch --tblout` format on the
original sequences, the same hits as `refined_hits.tsv`, the GPU candidate regions, the native
per-model `cmsearch` outputs and `summary.json`.

## Test

    INFERNAL_BIN=infernal-1.1.5/src python tests/test_infernal_gpu.py

The test (about a minute; `pytest tests` also works) calibrates two models and compares their
E-value thresholds with `cmcalibrate`, then searches a genome slice and compares every hit with
`cmsearch --nohmm`. Its data are files from the Infernal 1.1.5 tutorial.

## Citation and license

Please cite: [to be added]. Infernal-GPU is distributed under the BSD 3-Clause license (`LICENSE`).
