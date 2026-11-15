#ifndef INFERNAL_GPU_TYPES_H
#define INFERNAL_GPU_TYPES_H

#include <stdint.h>

/*
 * Flattened, GPU-friendly representation of an Infernal Covariance Model.
 * All nested pointer structures from CM_t are converted to flat 1D arrays
 * suitable for cudaMemcpy and coalesced GPU access.
 *
 * Naming convention:  d_xxx = device pointer,  h_xxx = host pointer.
 */

#define GPU_MAXCONNECT 6

/* Mirror of Infernal's state type constants */
#define GPU_D_st   0
#define GPU_MP_st  1
#define GPU_ML_st  2
#define GPU_MR_st  3
#define GPU_IL_st  4
#define GPU_IR_st  5
#define GPU_S_st   6
#define GPU_E_st   7
#define GPU_B_st   8
#define GPU_EL_st  9

/* Mirror of Infernal's state id constants */
#define GPU_BEGL_S  3
#define GPU_BEGR_S  4
#define GPU_BIF_B   19

/* Mirror of Infernal's emit mode constants */
#define GPU_EMITLEFT  0
#define GPU_EMITRIGHT 1
#define GPU_EMITPAIR  2
#define GPU_EMITNONE  3

/* Score constants matching Infernal */
#define GPU_IMPOSSIBLE (-1e36f)
#define GPU_NOT_IMPOSSIBLE(x) ((x) > -9.999e35f)

/* ---------- E-value calibration parameters ---------- */

typedef struct {
    float  mu;          /* mu_extrap from ECMLI/ECMLC line */
    float  lambda;      /* lambda from ECMLI/ECMLC line */
    double dbsize;      /* calibration database size (from ECMLI/ECMLC) */
    int    nrandhits;   /* number of random hits at calibration size */
    int    is_valid;    /* 1 if parameters were parsed from CM file */
} GPU_EvalueParams;

#define GPU_CM_NAME_MAX 256

/* ---------- Host-side CM structure (for staging before upload) ---------- */

typedef struct {
    int M;                /* number of states                            */
    int W;                /* max d (max hit size)                        */
    int abc_Kp;           /* alphabet size including degeneracies (typ. 18 for RNA) */
    float el_selfsc;      /* EL state self-transition score              */
    int has_local_begins; /* 1 if CMH_LOCAL_BEGIN flag is set            */

    /* Per-state arrays [M] */
    int8_t *sttype;       /* state type (D_st, MP_st, ..., EL_st)       */
    int8_t *stid;         /* unique state id (ROOT_S, BEGL_S, ...)      */
    int    *cfirst;       /* index of first child state                  */
    int    *cnum;         /* num children (or right child for B_st)      */
    int    *emitmode;     /* precomputed: EMITLEFT/RIGHT/PAIR/NONE       */
    int    *sd;           /* precomputed: StateDelta per state           */
    int    *sdr;          /* precomputed: StateRightDelta per state      */

    /* Transition scores [M * MAXCONNECT], row-major */
    float  *tsc;

    /* Emission scores: flat array + offset table */
    float  *oesc;         /* flat packed emission scores                 */
    int    *oesc_offset;  /* [M] byte offset into oesc for state v (-1 if none) */
    int     oesc_total;   /* total number of floats in oesc              */

    /* Local alignment scores [M] */
    float  *beginsc;
    float  *endsc;

    /* Precomputed init_sc = FCalcInitDPScores result [M * (W+1)], row-major */
    float  *init_sc;

    /* QDB bands [M] — per-state d-range limits (from CM file dmin1/dmax1) */
    int    *dmin1;        /* [M] QDB lower band (>= sd[v])               */
    int    *dmax1;        /* [M] QDB upper band (<= W)                   */

    /* BEGL_S state mapping */
    int     n_begl;       /* number of BEGL_S states                     */
    int    *begl_map;     /* [M] → index in BEGL_S array, or -1          */
    int    *begl_list;    /* [n_begl] → state index v for each BEGL_S    */

    /* E-value calibration (from CM file header EC* lines) */
    GPU_EvalueParams evp_cyk_local;       /* ECMLC */
    GPU_EvalueParams evp_ins_local;       /* ECMLI */

    /* Background and NULL3 prior (NULL / N3OMEGA lines) for Infernal's
     * composition correction; has_null3 = 0 if the lines were absent. */
    float null0[4];
    float null3_omega;
    int   has_null3;

    /* CM identity (for cmscan multi-CM mode) */
    char    name[GPU_CM_NAME_MAX];
    char    acc[GPU_CM_NAME_MAX];
} GPU_CM_Host;

/* ---------- Device-side pointers (after cudaMemcpy) ---------- */

/* Optional fully packed DP layout (kernel argument, by value). Each alpha row
 * stores only state v's band [a_lo[v], a_hi[v]] at a_off[v] (E states keep
 * d = 0); reads outside a band return IMPOSSIBLE, exactly what the dense
 * layout holds there. Each BEGL state b keeps a ring of b_ring[b] history
 * rows (its parent B reads j-k with k <= dmax of the BEGR sibling, root local
 * begins read k = 0) of width d_begl_width[b], starting at b_base[b]. */
typedef struct {
    int        full;
    int        a_width;       /* floats per alpha row */
    long long  begl_floats;   /* floats of all BEGL rings per window */
    const int *a_off, *a_lo, *a_hi;  /* [M] */
    const int *b_base, *b_ring;      /* [n_begl] */
} GPU_DPLayout;

typedef struct {
    int M;
    int W;
    int abc_Kp;
    float el_selfsc;
    int has_local_begins;
    int n_begl;

    /* Optional Inside-only BEGL d packing; all W+1 j slots remain. */
    int begl_packed_width;
    int *d_begl_offsets;
    int *d_begl_lo;
    int *d_begl_width;

    /* Device pointers */
    int8_t *d_sttype;
    int8_t *d_stid;
    int    *d_cfirst;
    int    *d_cnum;
    int    *d_emitmode;
    int    *d_sd;
    int    *d_sdr;

    float  *d_tsc;          /* [M * MAXCONNECT] */
    float  *d_oesc;         /* [oesc_total]     */
    int    *d_oesc_offset;  /* [M]              */

    float  *d_beginsc;      /* [M]              */
    float  *d_endsc;        /* [M]              */
    float  *d_init_sc;      /* [M * (W+1)]      */

    int    *d_begl_map;     /* [M]              */

    int    *d_dmin1;        /* [M] QDB lower band */
    int    *d_dmax1;        /* [M] QDB upper band */
    int il_nucleotide_emissions_zero; /* all IL scores zero for ACGU/IUPAC nucleotides */

    /* Optional dependency-level state schedule (STATE_LEVEL_SCHED builds).
     * Within one sequence position j, a state whose children are read from
     * row j-1 (sdr>0: MP/MR/IR) or from lower levels only can be evaluated
     * concurrently with the other states of its level. n_levels=0 selects the
     * legacy one-state-per-barrier loop. All arrays live in d_sched_buf. */
    int  sched_n_levels;
    int *d_sched_buf;
    const int *d_lvl_simple_begin;  /* [n_levels+1] -> index into simple_state */
    const int *d_simple_state;      /* non-IL states ordered by level          */
    const int *d_simple_item_start; /* [n_simple+1] cumulative band widths      */
    const int *d_lvl_il_begin;      /* [n_levels+1] -> index into il_state      */
    const int *d_il_state;          /* IL states ordered by level               */

    /* Fully packed layout metadata (device pointers inside); always built. */
    GPU_DPLayout full_layout;
    int *d_layout_buf;
} GPU_CM_Device;

/* Optional per-cell NULL3 correction of the root row, used only for the per-j
 * gate outputs (bestsc_j, dmin_j). Mirrors Infernal's ReportHitsGreedily:
 * corrected(j,d) = raw(j,d) - LogSum2(0, sum_a log2(c_a/null_a) c_a d + log2 omega)
 * where c is the Easel-counted composition of j-d+1..j. Cells with
 * raw < min_raw skip the correction (they cannot pass a gate >= min_raw).
 * act: per-window ring of (W+1) x 4 prefix counts, device scratch. */
typedef struct {
    int    enabled;
    float  min_raw;
    float  log2_omega;
    float  log2_null[4];
    float *act;
    /* Optional root-cell emission for E-value calibration (independent of
     * `enabled`): every root cell (window, end offset, length d) whose RAW
     * score is >= cell_cut is appended as 4 ints {window, j - i0, d, float
     * bits}. These fields ride in this struct because it already reaches
     * every kernel by value.
     * Caller side: h_cells has room for cell_cap cells and *h_cell_count is
     * the number stored so far; the launchers append and add the number of
     * qualifying cells, so a count above cell_cap signals overflow.
     * Device side (set by the launchers): cells, cell_count, cell_room. */
    int   *h_cells;
    int   *h_cell_count;
    int    cell_cap;
    float  cell_cut;
    int    cell_win_base;   /* index of the launch's first window in the caller's array */
    int   *cells;
    int   *cell_count;
    int    cell_room;
    /* CALIB_RECORDS builds: with cell_records set, a root cell is emitted only
     * if it can be reported by ReportHitsGreedily at its j, i.e. its NULL3-
     * corrected score exceeds every shorter length's corrected score minus
     * cell_margin (needs `enabled` for the composition ring). The margin
     * makes the set a superset under float differences; the host re-applies
     * the exact rule, so its hits equal those from emitting every cell. */
    int    cell_records;
    float  cell_margin;
} GPU_Null3;


/* Kernel-argument view of the level schedule (passed by value). */
typedef struct {
    int n_levels;
    const int *lvl_simple_begin;
    const int *simple_state;
    const int *simple_item_start;
    const int *lvl_il_begin;
    const int *il_state;
} GPU_LevelSched;

/* ---------- Batch scan workspace ---------- */

typedef struct {
    int full_layout;       /* 1: workspace uses the device's GPU_DPLayout */
    int begl_packed_width; /* 0: legacy dense BEGL; >0: packed row width */
    int max_batch;     /* maximum number of windows in one batch        */
    int M;             /* model size (states)                           */
    int W;             /* max window/hit size                           */
    int n_begl;        /* number of BEGL_S states                       */

    /* Per-window DP matrices in device global memory.
     * alpha:      [max_batch][2][M][W+1]       — rolling j, all non-BEGL_S states
     * alpha_begl: [max_batch][(W+1)][n_begl][W+1] — rolling j, BEGL_S states only
     */
    float *d_alpha;
    float *d_alpha_begl;

    /* Per-window auxiliary arrays */
    float *d_bestsc;   /* [max_batch][W+1]     — best score per d      */
    int   *d_bestr;    /* [max_batch][W+1]     — best root state per d */

    /* Per-window output */
    float *d_vsc_root; /* [max_batch] — best hit score per window       */
} GPU_ScanWorkspace;

/* ---------- Window descriptor (host-side) ---------- */

typedef struct {
    const uint8_t *dsq;   /* digitized sequence (1-based) */
    int64_t i0;           /* start position in dsq        */
    int64_t j0;           /* end position in dsq          */
    float   cutoff;       /* minimum bit score to report  */
} GPU_Window;

#endif /* INFERNAL_GPU_TYPES_H */
