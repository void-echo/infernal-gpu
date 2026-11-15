/* Generate the random sequences that Infernal's cmcalibrate searches by default:
 * N chunks of L residues sampled from its built-in genomic HMM, with the RNG
 * created and re-initialised exactly as cmcalibrate.c does (seed 181 by default).
 * cmcalibrate re-seeds before every CM, so every model is calibrated on this same set.
 * Usage: gen_calib_seqs [seed=181] [N=160] [L=10000] > calib_seqs.fa */
#include "esl_config.h"
#include "p7_config.h"
#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include "easel.h"
#include "esl_alphabet.h"
#include "esl_random.h"
#include "hmmer.h"
#include "infernal.h"

int main(int argc, char **argv)
{
  int seed = (argc > 1) ? atoi(argv[1]) : 181;
  int N    = (argc > 2) ? atoi(argv[2]) : 160;
  int L    = (argc > 3) ? atoi(argv[3]) : 10000;
  char errbuf[eslERRBUFSIZE];
  ESL_ALPHABET   *abc = esl_alphabet_Create(eslRNA);
  ESL_RANDOMNESS *r   = esl_randomness_Create(seed);
  double *sA = NULL, **tAA = NULL, **eAA = NULL;
  int nstates = 0, i, k;
  ESL_DSQ *dsq = NULL;

  if (CreateGenomicHMM(abc, errbuf, &sA, &tAA, &eAA, &nstates) != eslOK) { fprintf(stderr, "%s\n", errbuf); return 1; }
  esl_randomness_Init(r, esl_randomness_GetSeed(r));   /* as generate_sequences() does */
  for (i = 0; i < N; i++) {
    if (SampleGenomicSequenceFromHMM(r, abc, errbuf, sA, tAA, eAA, nstates, L, &dsq) != eslOK) { fprintf(stderr, "%s\n", errbuf); return 1; }
    printf(">calib%04d\n", i);
    for (k = 1; k <= L; k++) putchar(abc->sym[dsq[k]]);
    putchar('\n');
    free(dsq);
  }
  return 0;
}
