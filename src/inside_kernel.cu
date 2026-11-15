/*
 * inside_kernel.cu — CUDA kernel for Inside scanning DP on Covariance Models.
 *
 * This kernel mirrors Infernal 1.1.5's FastFInsideScan() in cm_dpsearch.c.
 * Structurally identical to cyk_kernel.cu, with the single algorithmic
 * change: all alternative-combining operations use logsum instead of max.
 *
 * logsum(a, b) = max(a,b) + log2(1 + 2^(min-max))
 *
 * Infernal scores are in bits (log base 2), so logsum uses log2/exp2
 * matching Infernal's FLogsum() in src/logsum.c which uses sreLOG2/sreEXP2.
 *
 * The final vsc_root (best hit score across all j, d) still uses max,
 * because we want the highest-scoring hit position.
 */

#include "gpu_types.h"
#include <math.h>

#if defined(STATE_LEVEL_SCHED) && (!defined(IL_PARALLEL_SCAN) || !defined(ROOT_STATE_FUSED) || defined(BIF_WARP_REDUCE))
#error "STATE_LEVEL_SCHED requires IL_PARALLEL_SCAN and ROOT_STATE_FUSED, without BIF_WARP_REDUCE"
#endif

/* ────────────── device helper functions ────────────── */

/*
 * Log-space addition matching Infernal's FLogsum() (src/logsum.c).
 * Scores are in bits (log base 2), so the correction uses log2/exp2:
 *   logsum(a, b) = max + log2(1 + 2^(min - max))
 * Cutoff at 23.0 matches Infernal (2^-23 ≈ FLT_EPSILON).
 */
__device__ __forceinline__
float gpu_logsum(float a, float b) {
    float mx = fmaxf(a, b);
    float mn = fminf(a, b);
    if (mn < GPU_IMPOSSIBLE + 1.0f) return mx;
    float diff = mx - mn;
    if (diff >= 23.0f) return mx;
    /* Use hardware SFU intrinsics for ~2x throughput vs IEEE-compliant versions.
     * __log2f: fast log2, ≤4 ULP error (~8 cycles via SFU vs ~20 for log2f).
     * exp2f(x) = expf(x*ln2): convert to __expf (fast natural exp, ≤2 ULP).
     * 0.6931472f = ln(2). Total error: ≤6 ULP ≈ 7e-7 bits — negligible
     * vs our 0.01-bit validation tolerance on genome sequences. */
    return mx + __log2f(1.0f + __expf((mn - mx) * 0.6931472f));
}

/* Semiring "addition": logsum for Inside, max for CYK. The CYK instantiation
 * shares every scheduling/storage optimization of the Inside kernel. */
template <bool MAXPLUS>
__device__ __forceinline__
float dp_combine(float a, float b) {
    return MAXPLUS ? fmaxf(a, b) : gpu_logsum(a, b);
}

__device__ __forceinline__
float gpu_max_f(float a, float b) {
    return fmaxf(a, b);
}

__host__ __device__ __forceinline__
int gpu_imax_ins(int a, int b) { return a > b ? a : b; }

__host__ __device__ __forceinline__
int gpu_imin_ins(int a, int b) { return a < b ? a : b; }

__host__ __device__ __forceinline__
int alpha_idx_ins(int M, int W1, int j_row, int v, int d) {
    return j_row * (M * W1) + v * W1 + d;
}

__device__ __forceinline__
int64_t alpha_begl_idx_ins(int n_begl, int W1, int j_slot, int begl_idx, int d,
                         int packed_width, const int *offsets, const int *lo) {
    if (packed_width > 0)
        return (int64_t)j_slot * packed_width + offsets[begl_idx] + d - lo[begl_idx];
    return (int64_t)j_slot * n_begl * W1 + begl_idx * W1 + d;
}

__device__ __forceinline__
float alpha_begl_load_ins(const float *alpha_begl, int n_begl, int W1, int j_slot,
                         int begl_idx, int d, int packed_width,
                         const int *offsets, const int *lo, const int *width) {
    if (packed_width > 0 && (d < lo[begl_idx] || d - lo[begl_idx] >= width[begl_idx]))
        return GPU_IMPOSSIBLE;
    return alpha_begl[alpha_begl_idx_ins(n_begl, W1, j_slot, begl_idx, d,
                                       packed_width, offsets, lo)];
}

/* Layout-generic accessors. L = 0 is the legacy layout (dense alpha rows,
 * optional packed BEGL d with W+1 ring rows) and compiles to the original
 * address arithmetic. L = 1 is GPU_DPLayout: packed alpha bands and short BEGL
 * rings; reads outside a band return IMPOSSIBLE, which is what the dense
 * layout holds in never-written cells. */
template <int L>
__device__ __forceinline__
int64_t a_idx(const GPU_DPLayout &ly, int M, int W1, int row, int v, int d) {
    if (L) return (int64_t)row * ly.a_width + ly.a_off[v] + d - ly.a_lo[v];
    return alpha_idx_ins(M, W1, row, v, d);
}

template <int L>
__device__ __forceinline__
float a_load(const float *alpha, const GPU_DPLayout &ly, int M, int W1, int row, int v, int d) {
    if (L && (d < ly.a_lo[v] || d > ly.a_hi[v])) return GPU_IMPOSSIBLE;
    return alpha[a_idx<L>(ly, M, W1, row, v, d)];
}

template <int L>
__device__ __forceinline__
int begl_ring(const GPU_DPLayout &ly, int b, int W1) {
    return L ? ly.b_ring[b] : W1;
}

template <int L>
__device__ __forceinline__
int begl_slot(const GPU_DPLayout &ly, int b, int64_t j, int W1) {
    return (int)(j % begl_ring<L>(ly, b, W1));
}

template <int L>
__device__ __forceinline__
int64_t b_idx(const GPU_DPLayout &ly, int n_begl, int W1, int slot, int b, int d,
              int packed_width, const int *offsets, const int *lo, const int *width) {
    if (L) return (int64_t)ly.b_base[b] + (int64_t)slot * width[b] + d - lo[b];
    return alpha_begl_idx_ins(n_begl, W1, slot, b, d, packed_width, offsets, lo);
}

template <int L>
__device__ __forceinline__
float b_load(const float *alpha_begl, const GPU_DPLayout &ly, int n_begl, int W1, int slot, int b, int d,
             int packed_width, const int *offsets, const int *lo, const int *width) {
    if (L) {
        if (d < lo[b] || d - lo[b] >= width[b]) return GPU_IMPOSSIBLE;
        return alpha_begl[b_idx<L>(ly, n_begl, W1, slot, b, d, packed_width, offsets, lo, width)];
    }
    return alpha_begl_load_ins(alpha_begl, n_begl, W1, slot, b, d, packed_width, offsets, lo, width);
}

#ifdef IL_PARALLEL_SCAN
/* ────── Warp/block prefix-scan helpers for IL_st parallelization ────── */

__device__ __forceinline__
float warp_inclusive_sum_scan(float v) {
    int lane = threadIdx.x & 31;
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        float prev = __shfl_up_sync(0xFFFFFFFF, v, offset);
        if (lane >= offset) v += prev;
    }
    return v;
}

template <bool MAXPLUS>
__device__ __forceinline__
float warp_inclusive_combine_scan(float v) {
    int lane = threadIdx.x & 31;
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        float prev = __shfl_up_sync(0xFFFFFFFF, v, offset);
        if (lane >= offset) v = dp_combine<MAXPLUS>(prev, v);
    }
    return v;
}

/* Block-wide inclusive prefix scans. Each thread provides one value; on
 * return, each thread receives the inclusive prefix over [v_0, ..., v_tid].
 * `shared_warp_tails` is a per-block scratch of >= n_warps floats.
 *
 * Block size must be a multiple of 32 (caller's responsibility).
 * Caller must NOT hold a __syncthreads barrier across this call:
 * the function performs its own internal __syncthreads.
 */
__device__ __forceinline__
float block_inclusive_sum_scan(float v, float *shared_warp_tails) {
    int tid     = threadIdx.x;
    int lane    = tid & 31;
    int warp_id = tid >> 5;
    int n_warps = (blockDim.x + 31) >> 5;

    float warp_scan = warp_inclusive_sum_scan(v);
    if (lane == 31) shared_warp_tails[warp_id] = warp_scan;
    __syncthreads();
    if (warp_id == 0) {
        float t = (lane < n_warps) ? shared_warp_tails[lane] : 0.0f;
        t = warp_inclusive_sum_scan(t);
        if (lane < n_warps) shared_warp_tails[lane] = t;
    }
    __syncthreads();
    if (warp_id > 0) warp_scan += shared_warp_tails[warp_id - 1];
    return warp_scan;
}

template <bool MAXPLUS>
__device__ __forceinline__
float block_inclusive_combine_scan(float v, float *shared_warp_tails) {
    int tid     = threadIdx.x;
    int lane    = tid & 31;
    int warp_id = tid >> 5;
    int n_warps = (blockDim.x + 31) >> 5;

    float warp_scan = warp_inclusive_combine_scan<MAXPLUS>(v);
    if (lane == 31) shared_warp_tails[warp_id] = warp_scan;
    __syncthreads();
    if (warp_id == 0) {
        float t = (lane < n_warps) ? shared_warp_tails[lane] : GPU_IMPOSSIBLE;
        t = warp_inclusive_combine_scan<MAXPLUS>(t);
        if (lane < n_warps) shared_warp_tails[lane] = t;
    }
    __syncthreads();
    if (warp_id > 0) warp_scan = dp_combine<MAXPLUS>(warp_scan, shared_warp_tails[warp_id - 1]);
    return warp_scan;
}

/* Float max via integer atomics: exact and order-independent (no NaNs occur;
 * the sign test also orders -0.0 below +0.0 like the unsigned-min branch). */
__device__ __forceinline__
void atomic_max_float(float *addr, float val) {
    /* Callers use this only on a row that a barrier-ordered store initialized
     * and that afterwards only increases, so a plain (possibly stale) read is
     * a lower bound on the current value: skipping when val <= it is exact.
     * Most local-begin candidates lose, so this avoids contended atomics. */
    if (val <= *(volatile float *)addr) return;
    if (!signbit(val)) atomicMax((int *)addr, __float_as_int(val));
    else               atomicMin((unsigned int *)addr, __float_as_uint(val));
}

/* One IL state for row j: alpha[v,d] = logsum(non_self[d], alpha[v,d-1]+t_self)
 * + esc[d], evaluated as a block-wide prefix scan (see derivation in the
 * kernel body history: h[k] = gamma[k] - CE[k] - (k-dn)*t_self, alpha[v,d] =
 * CE[d] + (d-dn)*t_self + prefix_logsum(h)). All threads must call this.
 * root_row != NULL additionally folds each written cell into the local-begin
 * root maximum (level-scheduled mode). */
template <bool MAXPLUS, int LY>
__device__ __forceinline__
void il_state_eval(const GPU_DPLayout &ly, int v, int64_t j, int cur, int dn, int dx, int first, int cn,
                   const float *tsc_v, const float *__restrict__ d_init_sc,
                   const float *__restrict__ d_oesc, int oesc_off, const uint8_t *dsq,
                   float *alpha, int M, int W1, int use_il_zero_emissions,
                   float *il_scan_tails_sum, float *il_scan_tails_lsum,
                   float *il_carry_ce, float *il_carry_pls,
                   float *root_row, float beginsc_v, int dmin0, int dmax0)
{
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int y = first;
    int yoff_self = -1;
    for (int yo = 0; yo < cn; yo++) {
        if (y + yo == v) { yoff_self = yo; break; }
    }
    float tsc_self = (yoff_self >= 0) ? tsc_v[yoff_self] : GPU_IMPOSSIBLE;
    bool has_self = (yoff_self >= 0) && GPU_NOT_IMPOSSIBLE(tsc_self);
    int n_d = dx - dn + 1;

    if (n_d <= 0) {
        /* no-op */
    } else if (!has_self) {
        for (int d_off = tid; d_off < n_d; d_off += nthreads) {
            int d = dn + d_off;
            int dp_y = d - 1;
            float sc = d_init_sc[v * W1 + dp_y];
            for (int yoff = 0; yoff < cn; yoff++)
                sc = dp_combine<MAXPLUS>(sc, a_load<LY>(alpha, ly, M, W1, cur, y + yoff, dp_y) + tsc_v[yoff]);
            int i = (int)(j - d + 1);
            float esc_d = (!use_il_zero_emissions && oesc_off >= 0) ? d_oesc[oesc_off + dsq[i]] : 0.0f;
            float val = sc + esc_d;
            alpha[a_idx<LY>(ly, M, W1, cur, v, d)] = val;
            if (root_row && d >= dmin0 && d <= dmax0) atomic_max_float(&root_row[d], val + beginsc_v);
        }
    } else {
        if (tid == 0) { *il_carry_ce = 0.0f; *il_carry_pls = GPU_IMPOSSIBLE; }
        __syncthreads();

        for (int tile_start = 0; tile_start < n_d; tile_start += nthreads) {
            /* Race fix (2026-07-14): this barrier + the one after the
             * alpha write below close a block-sync gap in the IL_PAR
             * tile loop. Benign on sm_86 but corrupts scores on sm_80
             * (A100) at high block occupancy.
             * Value-neutral: sm_86 output is byte-identical with/without. */
            __syncthreads();
            int d_off = tile_start + tid;
            int d     = dn + d_off;
            bool active = (d_off < n_d);

            float gamma = GPU_IMPOSSIBLE;
            float esc_d = 0.0f;
            if (active) {
                int dp_y = d - 1;
                float sc = d_init_sc[v * W1 + dp_y];
                for (int yoff = 0; yoff < cn; yoff++) {
                    if (yoff == yoff_self) continue;
                    sc = dp_combine<MAXPLUS>(sc, a_load<LY>(alpha, ly, M, W1, cur, y + yoff, dp_y) + tsc_v[yoff]);
                }
                int i = (int)(j - d + 1);
                esc_d = (!use_il_zero_emissions && oesc_off >= 0) ? d_oesc[oesc_off + dsq[i]] : 0.0f;
                gamma = sc + esc_d;
            }

            float my_esc = active ? esc_d : 0.0f;
            float ce = 0.0f;
            if (!use_il_zero_emissions) {
                float ce_in_tile = block_inclusive_sum_scan(my_esc, il_scan_tails_sum);
                ce = ce_in_tile + *il_carry_ce;
            }

#ifdef IL_ZERO_EMISSION_SCAN
            /* Preserve the legacy shared-product rounding. Without
             * explicit operations, ptxas may contract the zero-CE
             * variant differently even when its runtime guard is off. */
            float self_term = __fmul_rn((float)d_off, tsc_self);
            float h = active ? __fsub_rn(__fsub_rn(gamma, ce), self_term) : GPU_IMPOSSIBLE;
#else
            float h = active ? (gamma - ce - (float)d_off * tsc_self) : GPU_IMPOSSIBLE;
#endif

            float pls_in_tile = block_inclusive_combine_scan<MAXPLUS>(h, il_scan_tails_lsum);
            float pls = dp_combine<MAXPLUS>(pls_in_tile, *il_carry_pls);

            if (active) {
#ifdef IL_ZERO_EMISSION_SCAN
                float val = __fadd_rn(__fadd_rn(ce, self_term), pls);
#else
                float val = ce + (float)d_off * tsc_self + pls;
#endif
                alpha[a_idx<LY>(ly, M, W1, cur, v, d)] = val;
                if (root_row && d >= dmin0 && d <= dmax0) atomic_max_float(&root_row[d], val + beginsc_v);
            }
            __syncthreads();  /* race fix — see tile-loop head above */

            int last_active_local = nthreads - 1;
            int remaining = n_d - tile_start;
            if (remaining < nthreads) last_active_local = remaining - 1;
            if (tid == last_active_local) {
                if (!use_il_zero_emissions) *il_carry_ce = ce;
                *il_carry_pls = pls;
            }
            __syncthreads();
        }
    }
}
#endif /* IL_PARALLEL_SCAN */

#ifdef BIF_WARP_REDUCE
/* ────── Warp logsum reduction for bifurcation k-loop ──────
 *
 * Bifurcation (B_st) work per cell is O(d): a serial inner logsum over the
 * split point k in [kmin, kmax]. The default scheme parallelises over d
 * across all block threads, leaving each thread to walk its k-range serially.
 * Here we instead give each d to one warp: the 32 lanes split the k-range
 * (lane handles k = kmin+lane, kmin+lane+32, …), accumulate partial logsums,
 * then a tree reduction combines them. This (a) keeps threads busy when the
 * d-band is narrower than the block (load balance), and (b) shortens the
 * dependent logsum chain from |k-range| to |k-range|/32 + log2(32), helping
 * latency at the low occupancies typical of large-W CMs. Assumes blockDim.x
 * is a multiple of 32 (same precondition as the IL_PAR scans). Result lands
 * in lane 0.
 */
template <bool MAXPLUS>
__device__ __forceinline__
float warp_logsum_reduce(float v) {
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, v, offset);
        v = dp_combine<MAXPLUS>(v, other);
    }
    return v;
}
#endif /* BIF_WARP_REDUCE */


/* Easel esl_abc_DCount weights for the RNA alphabet ACGU-RYMKSWHBVDN:
 * canonical residues count 1, degenerate codes split 1 evenly over their
 * canonical members, gap/missing/nonresidue are not counted (a gap never
 * occurs in scanned targets). */
__device__ __forceinline__
void easel_count_rna(uint8_t x, float w[4]) {
    const unsigned masks[16] = { 1, 2, 4, 8, 0, 5, 10, 3, 12, 6, 9, 11, 14, 7, 13, 15 };
    w[0] = w[1] = w[2] = w[3] = 0.0f;
    if (x > 15) return;
    unsigned m = masks[x];
    float n = (float)__popc(m);
    if (n == 0.0f) return;
    for (int a = 0; a < 4; a++) if (m & (1u << a)) w[a] = 1.0f / n;
}

/* Infernal ScoreCorrectionNull3 for composition counts c (unnormalized).
 * Out of line: it runs only for root cells above a raw gate, and inlining
 * its transcendental math into the DP loop raised register use enough to
 * cost resident blocks (measured 8 -> 5 per SM for CYK on sm_86). */
__device__ __noinline__
float null3_correction(const float *hi, const float *lo, int d, GPU_Null3 n3) {
    float c[4], tot = 0.0f;
    for (int a = 0; a < 4; a++) { c[a] = hi[a] - lo[a]; if (c[a] < 0.0f) c[a] = 0.0f; tot += c[a]; }
    float score = n3.log2_omega;
    for (int a = 0; a < 4; a++) {
        float comp = (tot > 0.0f) ? c[a] / tot : 0.25f;          /* esl_vec_FNorm */
        if (comp > 0.0f) score += (log2f(comp) - n3.log2_null[a]) * comp * (float)d;
    }
    float mx = fmaxf(0.0f, score), mn = fminf(0.0f, score);  /* LogSum2(0, score) */
    return (mx - mn >= 50.0f) ? mx : mx + log2f(1.0f + exp2f(mn - mx));
}

/* ────────────── Main Inside Scan Kernel ────────────── */

/* Optional occupancy cap: INSIDE_MIN_BLOCKS_256=k asks ptxas to fit k resident
 * 256-thread blocks per SM (i.e. 2k of the default 128-thread blocks), which
 * bounds registers per thread. Unset keeps the compiler's choice. */
#ifdef INSIDE_MIN_BLOCKS_256
#define INSIDE_LAUNCH_BOUNDS __launch_bounds__(256, INSIDE_MIN_BLOCKS_256)
#else
#define INSIDE_LAUNCH_BOUNDS
#endif

template <bool MAXPLUS, int LY>
__device__ __forceinline__ void
dp_scan_body(
    /* CM (read-only) */
    int         cm_M,
    int         cm_W,
    int         cm_abc_Kp,
    float       cm_el_selfsc,
    int         cm_has_local_begins,
    int         cm_n_begl,
    const int8_t  *__restrict__ d_sttype,
    const int8_t  *__restrict__ d_stid,
    const int     *__restrict__ d_cfirst,
    const int     *__restrict__ d_cnum,
    const int     *__restrict__ d_emitmode,
    const int     *__restrict__ d_sd,
    const int     *__restrict__ d_sdr,
    const float   *__restrict__ d_tsc,
    const float   *__restrict__ d_oesc,
    const int     *__restrict__ d_oesc_offset,
    const float   *__restrict__ d_beginsc,
    const float   *__restrict__ d_init_sc,
    const int     *__restrict__ d_begl_map,
    const int     *__restrict__ d_dmin1,
    const int     *__restrict__ d_dmax1,
    /* Batch windows */
    const uint8_t *__restrict__ d_dsq_batch,
    const int64_t *__restrict__ d_dsq_offset,
    const int64_t *__restrict__ d_win_i0,
    const int64_t *__restrict__ d_win_j0,
    int         n_windows,
    /* Workspace (pre-allocated per window) */
    float       *d_alpha,
    float       *d_alpha_begl,
    /* Output */
    float       *d_vsc_root,
    /* Optional per-end-position output: bestsc[j] = best local-Inside hit
     * score ENDING at position j (max over all lengths d and root local
     * begins). This is exactly the per-j input cmcalibrate feeds to its gamma
     * semi-HMM hit resolver. NULL => skip (legacy behavior). Stride = perj_stride
     * floats per window; bestsc[j] written at d_bestsc_j[win*perj_stride + (j-i0)]. */
    float       *d_bestsc_j,
    int          perj_stride,
    /* Optional FULL per-(j,d) root score output: alpha0[j][d] = best local-Inside
     * score of a hit of length d ending at j (this is cmcalibrate's per-length
     * bestsc[d], fed to UpdateGammaHitMx for ALL d). NULL => skip. Layout:
     * d_alpha0_jd[((int64)win*perj_stride + (j-i0))*alpha0_dstride + d]. */
    float       *d_alpha0_jd,
    int          alpha0_dstride,
    int          use_il_zero_emissions,
    int          begl_packed_width,
    const int *d_begl_offsets, const int *d_begl_lo, const int *d_begl_width,
    GPU_LevelSched sched,
    /* Optional per-j minimum root length d with score >= dmin_cut (NULL =>
     * skip; 0x7fffffff when none). Implemented for ROOT_STATE_FUSED builds;
     * other builds leave 0x7fffffff, which callers must treat as unknown. */
    float        dmin_cut,
    int         *d_dmin_j,
    GPU_Null3    n3,
    GPU_DPLayout ly
)
{
#ifndef IL_ZERO_EMISSION_SCAN
    use_il_zero_emissions = 0;
#endif
    int win = blockIdx.x;
    if (win >= n_windows) return;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

#ifdef IL_PARALLEL_SCAN
    /* Scratch for block-wide prefix scans used in parallel IL_st path.
     * Sized for blockDim.x up to 256 (8 warps). Trivial bytes. */
    __shared__ float il_scan_tails_sum[8];
    __shared__ float il_scan_tails_lsum[8];
    __shared__ float il_carry_ce;
    __shared__ float il_carry_pls;
#endif

    int M  = cm_M;
    int Kp = cm_abc_Kp;

    int W_max = cm_W;
    int W1_max = W_max + 1;
    int64_t alpha_win_stride      = LY ? (int64_t)2 * ly.a_width : (int64_t)2 * M * W1_max;
    int64_t alpha_begl_win_stride = LY ? (int64_t)ly.begl_floats
                                      : (int64_t)W1_max * (begl_packed_width > 0 ? begl_packed_width : cm_n_begl * W1_max);
    float *alpha      = d_alpha      + (int64_t)win * alpha_win_stride;
    float *alpha_begl = d_alpha_begl + (int64_t)win * alpha_begl_win_stride;

    int64_t i0 = d_win_i0[win];
    int64_t j0 = d_win_j0[win];
    int64_t L  = j0 - i0 + 1;
    int W  = (W_max > (int)L) ? (int)L : W_max;

    const int W1 = W1_max;
    const uint8_t *dsq = d_dsq_batch + d_dsq_offset[win];

    /* Initialize alpha and alpha_begl to IMPOSSIBLE */
    for (int64_t i = tid; i < alpha_win_stride; i += nthreads)
        alpha[i] = GPU_IMPOSSIBLE;
    for (int64_t i = tid; i < alpha_begl_win_stride; i += nthreads)
        alpha_begl[i] = GPU_IMPOSSIBLE;
    __syncthreads();

    /* E_st base case */
    for (int v = tid; v < M; v += nthreads) {
        if (d_sttype[v] == GPU_E_st) {
            alpha[a_idx<LY>(ly, M, W1_max, 0, v, 0)] = 0.0f;
            alpha[a_idx<LY>(ly, M, W1_max, 1, v, 0)] = 0.0f;
        }
    }
    __syncthreads();

    __shared__ int s_dmin;
    if (tid == 0) s_dmin = 0x7fffffff;
    float *n3_act = n3.enabled ? n3.act + (int64_t)win * W1_max * 4 : NULL;
    if (n3_act && tid < 4) n3_act[tid] = 0.0f;        /* composition before i0 */
    __syncthreads();

    float vsc_root = GPU_IMPOSSIBLE;

    for (int64_t j = i0; j <= j0; j++) {
        int cur  = (int)(j % 2);
        int prv  = (int)((j - 1) % 2);
        int jp_g = (int)(j - i0 + 1);
        if (n3_act && tid == 0) {
            /* Prefix composition ring (Infernal's act[]); read only in the root
             * section below, after the state loop's barriers. */
            float w[4];
            easel_count_rna(dsq[j], w);
            const float *prev = n3_act + ((jp_g - 1) % W1_max) * 4;
            float *curc = n3_act + (jp_g % W1_max) * 4;
            for (int a = 0; a < 4; a++) curc[a] = prev[a] + w[a];
        }
#ifdef BIF_RING_INDEX
        /* Valid QDB k is in [0,W], so (j-k) mod (W+1) needs at most
         * one wrap after subtracting k from the precomputed j slot. */
        int j_begl_slot = (int)(j % W1);
#endif

#ifdef ROOT_STATE_FUSED
        if (cm_has_local_begins) {
            for (int d = d_dmin1[0] + tid; d <= gpu_imin_ins(d_dmax1[0], jp_g); d += nthreads)
                alpha[a_idx<LY>(ly, M, W1, cur, 0, d)] = GPU_IMPOSSIBLE;
            __syncthreads();
        }
#endif
#ifdef STATE_LEVEL_SCHED
        if (LY == 0 && sched.n_levels > 0) {
            /* Dependency-level schedule: all states in one level are mutually
             * independent for this j, so their (state,d) cells form one flat
             * parallel loop with a single barrier per level. Per-cell
             * arithmetic and child order are those of the legacy loop. */
            float *root_row = alpha + a_idx<LY>(ly, M, W1, cur, 0, 0);
            int dmin0 = d_dmin1[0], dmax0 = d_dmax1[0];
            for (int l = 0; l < sched.n_levels; l++) {
                int s0 = sched.lvl_simple_begin[l], s1 = sched.lvl_simple_begin[l + 1];
                int it1 = sched.simple_item_start[s1];
                int k = s0;
                for (int it = sched.simple_item_start[s0] + tid; it < it1; it += nthreads) {
                    while (sched.simple_item_start[k + 1] <= it) k++;
                    int v = sched.simple_state[k];
                    int d = d_dmin1[v] + (it - sched.simple_item_start[k]);
                    if (d > jp_g) continue;
                    int st    = d_sttype[v];
                    int first = d_cfirst[v];
                    int cn    = d_cnum[v];
                    int jp_y  = (d_sdr[v] > 0) ? prv : cur;
                    float val;
                    if (st == GPU_B_st) {
                        int w = first, y = cn;
                        int begl_w = d_begl_map[w];
                        int kmin = gpu_imax_ins(d_dmin1[y], d - d_dmax1[w]);
                        int kmax = gpu_imin_ins(d_dmax1[y], d - d_dmin1[w]);
                        float sc = d_init_sc[v * W1 + d];
                        for (int kk = kmin; kk <= kmax; kk++) {
#ifdef BIF_RING_INDEX
                            int jp_wk = (LY ? begl_slot<LY>(ly, begl_w, j, W1) : j_begl_slot) - kk;
#else
                            int jp_wk = (int)((j - kk) % W1);
#endif
                            if (jp_wk < 0) jp_wk += begl_ring<LY>(ly, begl_w, W1);
                            float left  = b_load<LY>(alpha_begl, ly, cm_n_begl, W1, jp_wk, begl_w, d - kk, begl_packed_width, d_begl_offsets, d_begl_lo, d_begl_width);
                            float right = alpha[a_idx<LY>(ly, M, W1, jp_y, y, kk)];
                            sc = dp_combine<MAXPLUS>(sc, left + right);
                        }
                        val = sc;
                        alpha[a_idx<LY>(ly, M, W1, cur, v, d)] = val;
                    } else {
                        int sd_v = d_sd[v];
                        const float *tsc_v = d_tsc + v * GPU_MAXCONNECT;
                        float sc = d_init_sc[v * W1 + d - sd_v];
                        for (int yoff = 0; yoff < cn; yoff++)
                            sc = dp_combine<MAXPLUS>(sc, a_load<LY>(alpha, ly, M, W1, jp_y, first + yoff, d - sd_v) + tsc_v[yoff]);
                        if (d_stid[v] == GPU_BEGL_S) {
                            val = sc;
                            alpha_begl[b_idx<LY>(ly, cm_n_begl, W1, begl_slot<LY>(ly, d_begl_map[v], j, W1), d_begl_map[v], d, begl_packed_width, d_begl_offsets, d_begl_lo, d_begl_width)] = val;
                        } else {
                            int em  = d_emitmode[v];
                            int off = d_oesc_offset[v];
                            float esc = 0.0f;
                            if (em == GPU_EMITLEFT) {
                                esc = (off >= 0) ? d_oesc[off + dsq[j - d + 1]] : 0.0f;
                            } else if (em == GPU_EMITRIGHT) {
                                esc = (off >= 0) ? d_oesc[off + dsq[j]] : GPU_IMPOSSIBLE;
                            } else if (em == GPU_EMITPAIR) {
                                esc = (off >= 0) ? d_oesc[off + dsq[j - d + 1] * Kp + dsq[j]] : 0.0f;
                            }
                            val = sc + esc;
                            alpha[a_idx<LY>(ly, M, W1, cur, v, d)] = val;
                        }
                    }
                    if (cm_has_local_begins && d >= dmin0 && d <= dmax0) {
                        float begin = d_beginsc[v];
                        if (GPU_NOT_IMPOSSIBLE(begin)) atomic_max_float(&root_row[d], val + begin);
                    }
                }
                for (int q = sched.lvl_il_begin[l]; q < sched.lvl_il_begin[l + 1]; q++) {
                    int v = sched.il_state[q];
                    float begin = d_beginsc[v];
                    bool fold = cm_has_local_begins && GPU_NOT_IMPOSSIBLE(begin);
                    il_state_eval<MAXPLUS, LY>(ly, v, j, cur, d_dmin1[v], gpu_imin_ins(d_dmax1[v], jp_g),
                                  d_cfirst[v], d_cnum[v], d_tsc + v * GPU_MAXCONNECT, d_init_sc, d_oesc,
                                  d_oesc_offset[v], dsq, alpha, M, W1, use_il_zero_emissions,
                                  il_scan_tails_sum, il_scan_tails_lsum, &il_carry_ce, &il_carry_pls,
                                  fold ? root_row : NULL, begin, dmin0, dmax0);
                }
                __syncthreads();
            }
        } else
#endif
        for (int v = M - 1; v > 0; v--) {
            int st = d_sttype[v];
            if (st == GPU_E_st) continue;

            int sid       = d_stid[v];
            int em        = d_emitmode[v];
            int sd_v      = d_sd[v];
            int sdr_v     = d_sdr[v];
            int first     = d_cfirst[v];
            int cn        = d_cnum[v];
            int jp_v      = (sid == GPU_BEGL_S) ? begl_slot<LY>(ly, d_begl_map[v], j, W1) : cur;
            int jp_y      = (sdr_v > 0) ? prv : cur;

            int dn = d_dmin1[v];
            int dx = gpu_imin_ins(d_dmax1[v], jp_g);

            float esc_j = GPU_IMPOSSIBLE;
            if (st == GPU_IR_st || st == GPU_MR_st) {
                int off = d_oesc_offset[v];
                if (off >= 0) esc_j = d_oesc[off + dsq[j]];
            }
            const float *tsc_v = d_tsc + v * GPU_MAXCONNECT;

            if (st == GPU_B_st) {
                int w = first;
                int y = cn;
                int begl_w = d_begl_map[w];
                int dmin1_w = d_dmin1[w];
                int dmax1_w = d_dmax1[w];
                int dmin1_y = d_dmin1[y];
                int dmax1_y = d_dmax1[y];

#ifdef BIF_WARP_REDUCE
                /* Warp-cooperative k-reduction: one warp per d, 32 lanes split
                 * the k-range, then warp_logsum_reduce combines partials.
                 * Lane 0 seeds with init_sc; others seed IMPOSSIBLE so the
                 * reduction yields logsum(init_sc, all left+right terms),
                 * identical (modulo float reordering) to the serial loop. */
                int lane    = tid & 31;
                int warp_id = tid >> 5;
                int n_warps = nthreads >> 5;
                for (int d = dn + warp_id; d <= dx; d += n_warps) {
                    int kmin = gpu_imax_ins(dmin1_y, d - dmax1_w);
                    int kmax = gpu_imin_ins(dmax1_y, d - dmin1_w);
                    float sc = (lane == 0) ? d_init_sc[v * W1 + d] : GPU_IMPOSSIBLE;
                    for (int k = kmin + lane; k <= kmax; k += 32) {
#ifdef BIF_RING_INDEX
                        int jp_wk = (LY ? begl_slot<LY>(ly, begl_w, j, W1) : j_begl_slot) - k;
#else
                        int jp_wk = (int)((j - k) % W1);
#endif
                        if (jp_wk < 0) jp_wk += begl_ring<LY>(ly, begl_w, W1);
                        float left  = b_load<LY>(alpha_begl, ly, cm_n_begl, W1, jp_wk, begl_w, d - k, begl_packed_width, d_begl_offsets, d_begl_lo, d_begl_width);
                        float right = alpha[a_idx<LY>(ly, M, W1, jp_y, y, k)];
                        sc = dp_combine<MAXPLUS>(sc, left + right);
                    }
                    sc = warp_logsum_reduce<MAXPLUS>(sc);
                    if (lane == 0)
                        alpha[a_idx<LY>(ly, M, W1, jp_v, v, d)] = sc;
                }
#else
                for (int d = dn + tid; d <= dx; d += nthreads) {
                    int kmin = gpu_imax_ins(dmin1_y, d - dmax1_w);
                    int kmax = gpu_imin_ins(dmax1_y, d - dmin1_w);
                    float sc = d_init_sc[v * W1 + d];
                    for (int k = kmin; k <= kmax; k++) {
#ifdef BIF_RING_INDEX
                        int jp_wk = (LY ? begl_slot<LY>(ly, begl_w, j, W1) : j_begl_slot) - k;
#else
                        int jp_wk = (int)((j - k) % W1);
#endif
                        if (jp_wk < 0) jp_wk += begl_ring<LY>(ly, begl_w, W1);
                        float left  = b_load<LY>(alpha_begl, ly, cm_n_begl, W1, jp_wk, begl_w, d - k, begl_packed_width, d_begl_offsets, d_begl_lo, d_begl_width);
                        float right = alpha[a_idx<LY>(ly, M, W1, jp_y, y, k)];
                        sc = dp_combine<MAXPLUS>(sc, left + right);
                    }
                    alpha[a_idx<LY>(ly, M, W1, jp_v, v, d)] = sc;
                }
#endif
            }
            else if (sid == GPU_BEGL_S) {
                int begl_v = d_begl_map[v];
                int y = first;
                for (int d = dn + tid; d <= dx; d += nthreads) {
                    float sc = d_init_sc[v * W1 + d];
                    for (int yoff = 0; yoff < cn; yoff++) {
                        sc = dp_combine<MAXPLUS>(sc, a_load<LY>(alpha, ly, M, W1, jp_y, y + yoff, d - sd_v) + tsc_v[yoff]);
                    }
                    alpha_begl[b_idx<LY>(ly, cm_n_begl, W1, jp_v, begl_v, d, begl_packed_width, d_begl_offsets, d_begl_lo, d_begl_width)] = sc;
                }
            }
            else if (st == GPU_IL_st) {
#ifdef IL_PARALLEL_SCAN
                /* IL_st parallel path: prefix-scan reformulation of the
                 * insert-left self loop; see il_state_eval<MAXPLUS, LY>(ly, ). */
                il_state_eval<MAXPLUS, LY>(ly, v, j, cur, dn, dx, first, cn, tsc_v, d_init_sc, d_oesc,
                              d_oesc_offset[v], dsq, alpha, M, W1, use_il_zero_emissions,
                              il_scan_tails_sum, il_scan_tails_lsum, &il_carry_ce, &il_carry_pls,
                              NULL, 0.0f, 0, -1);
#else
                /* Original serial IL_st (tid==0 due to self-loop dependency) */
                if (tid == 0) {
                    int y = first;
                    for (int d = dn; d <= dx; d++) {
                        int dp_y = d - sd_v;
                        float sc = d_init_sc[v * W1 + dp_y];
                        for (int yoff = 0; yoff < cn; yoff++)
                            sc = dp_combine<MAXPLUS>(sc, a_load<LY>(alpha, ly, M, W1, jp_y, y + yoff, dp_y) + tsc_v[yoff]);
                        int i = (int)(j - d + 1);
                        int off = d_oesc_offset[v];
                        sc += (off >= 0) ? d_oesc[off + dsq[i]] : 0.0f;
                        alpha[a_idx<LY>(ly, M, W1, jp_v, v, d)] = sc;
                    }
                }
#endif
            }
            else if (st == GPU_IR_st) {
                int y = first;
                for (int d = dn + tid; d <= dx; d += nthreads) {
                    int dp_y = d - sd_v;
                    float sc = d_init_sc[v * W1 + dp_y];
                    for (int yoff = 0; yoff < cn; yoff++)
                        sc = dp_combine<MAXPLUS>(sc, a_load<LY>(alpha, ly, M, W1, jp_y, y + yoff, dp_y) + tsc_v[yoff]);
                    sc += esc_j;
                    alpha[a_idx<LY>(ly, M, W1, jp_v, v, d)] = sc;
                }
            }
            else {
                int y = first;
                for (int d = dn + tid; d <= dx; d += nthreads) {
                    int dp_y = d - sd_v;
                    float sc = d_init_sc[v * W1 + dp_y];
                    for (int yoff = 0; yoff < cn; yoff++) {
                        sc = dp_combine<MAXPLUS>(sc, a_load<LY>(alpha, ly, M, W1, jp_y, y + yoff, dp_y) + tsc_v[yoff]);
                    }
                    float esc = 0.0f;
                    if (em == GPU_EMITLEFT) {
                        int i = (int)(j - d + 1);
                        int off = d_oesc_offset[v];
                        esc = (off >= 0) ? d_oesc[off + dsq[i]] : 0.0f;
                    } else if (em == GPU_EMITRIGHT) {
                        esc = esc_j;
                    } else if (em == GPU_EMITPAIR) {
                        int i = (int)(j - d + 1);
                        int off = d_oesc_offset[v];
                        esc = (off >= 0) ? d_oesc[off + dsq[i] * Kp + dsq[j]] : 0.0f;
                    }
                    alpha[a_idx<LY>(ly, M, W1, jp_v, v, d)] = sc + esc;
                }
            }

#ifdef ROOT_STATE_FUSED
            /* Reuse state-completion synchronization instead of making a
             * second traversal with one barrier per local begin. Iteration
             * starts at dn (not max(dn,dn0)) so the writer of state(v,d) also
             * reads it here. Alternate B/IL writers need the barrier below. */
#ifdef BIF_WARP_REDUCE
            if (st == GPU_B_st) __syncthreads();
#endif
#ifndef IL_PARALLEL_SCAN
            if (st == GPU_IL_st) __syncthreads();
#endif
            if (cm_has_local_begins && GPU_NOT_IMPOSSIBLE(d_beginsc[v])) {
                for (int d = dn + tid; d <= dx; d += nthreads) {
                    if (d < d_dmin1[0] || d > d_dmax1[0]) continue;
                    float child = (sid == GPU_BEGL_S)
                        ? b_load<LY>(alpha_begl, ly, cm_n_begl, W1, jp_v, d_begl_map[v], d, begl_packed_width, d_begl_offsets, d_begl_lo, d_begl_width)
                        : alpha[a_idx<LY>(ly, M, W1, cur, v, d)];
                    int64_t root_idx = a_idx<LY>(ly, M, W1, cur, 0, d);
                    alpha[root_idx] = fmaxf(alpha[root_idx], child + d_beginsc[v]);
                }
            }
#endif
            __syncthreads();
        } /* end v-loop */

        __syncthreads();

        /* ──── Handle ROOT (v=0) and local begins ──── */
        {
            const float *tsc_0 = d_tsc;
            int y0    = d_cfirst[0];
            int cn0   = d_cnum[0];
            int dn0   = d_dmin1[0];
            int dx0   = gpu_imin_ins(d_dmax1[0], jp_g);

#if defined(ROOT_STATE_FUSED)
            float local_max = GPU_IMPOSSIBLE;
#ifdef CALIB_RECORDS
            if (n3.cells != NULL && n3.cell_records) {
                /* Calibration: lengths are visited in ascending blocks of
                 * nthreads; a block-wide prefix maximum of the corrected
                 * scores (carried across blocks) identifies the cells whose
                 * corrected score is a new running maximum over shorter
                 * lengths, which are the only cells ReportHitsGreedily can
                 * report at this j. Requires nthreads to be a multiple of 32. */
                __shared__ float s_wmax[32];
                __shared__ float s_carry;
                const int lane = tid & 31, wid = tid >> 5, nw = nthreads >> 5;
                if (tid == 0) s_carry = GPU_IMPOSSIBLE;
                __syncthreads();
                for (int base = dn0; base <= dx0; base += nthreads) {
                    const int d = base + tid;
                    float sc = GPU_IMPOSSIBLE, c = GPU_IMPOSSIBLE;
                    if (d <= dx0) {
                        sc = dp_combine<MAXPLUS>(sc, a_load<LY>(alpha, ly, M, W1, cur, y0, d) + tsc_0[0]);
                        for (int yoff = 1; yoff < cn0; yoff++)
                            sc = dp_combine<MAXPLUS>(sc, a_load<LY>(alpha, ly, M, W1, cur, y0 + yoff, d) + tsc_0[yoff]);
                        int64_t root_idx = a_idx<LY>(ly, M, W1, cur, 0, d);
                        if (cm_has_local_begins) sc = fmaxf(sc, alpha[root_idx]);
                        alpha[root_idx] = sc;
                        c = sc;
                        if (n3_act && GPU_NOT_IMPOSSIBLE(sc))
                            c = sc - null3_correction(n3_act + (jp_g % W1_max) * 4,
                                                      n3_act + ((jp_g - d) % W1_max) * 4, d, n3);
                        if (d < 1) c = GPU_IMPOSSIBLE;
                        local_max = gpu_max_f(local_max, c);
                    }
                    float v = c;
                    for (int o = 1; o < 32; o <<= 1) {
                        float t = __shfl_up_sync(0xffffffffu, v, o);
                        if (lane >= o) v = fmaxf(v, t);
                    }
                    const float prev_lane = __shfl_up_sync(0xffffffffu, v, 1);
                    if (lane == 31) s_wmax[wid] = v;
                    __syncthreads();
                    float excl = s_carry;
                    for (int w = 0; w < wid; w++) excl = fmaxf(excl, s_wmax[w]);
                    if (lane > 0) excl = fmaxf(excl, prev_lane);
                    if (d <= dx0 && GPU_NOT_IMPOSSIBLE(c) && c > excl - n3.cell_margin && sc >= n3.cell_cut) {
                        int slot = atomicAdd(n3.cell_count, 1);
                        if (slot < n3.cell_room) {
                            int *cell = n3.cells + 4 * (int64_t)slot;
                            cell[0] = n3.cell_win_base + win;
                            cell[1] = (int)(j - i0);
                            cell[2] = d;
                            cell[3] = __float_as_int(sc);
                        }
                    }
                    __syncthreads();
                    if (tid == 0) {
                        float t = s_carry;
                        for (int w = 0; w < nw; w++) t = fmaxf(t, s_wmax[w]);
                        s_carry = t;
                    }
                    __syncthreads();
                }
            } else
#endif
            for (int d = dn0 + tid; d <= dx0; d += nthreads) {
                float sc = GPU_IMPOSSIBLE;
                sc = dp_combine<MAXPLUS>(sc, a_load<LY>(alpha, ly, M, W1, cur, y0, d) + tsc_0[0]);
                for (int yoff = 1; yoff < cn0; yoff++) {
                    sc = dp_combine<MAXPLUS>(sc, a_load<LY>(alpha, ly, M, W1, cur, y0 + yoff, d) + tsc_0[yoff]);
                }
                int64_t root_idx = a_idx<LY>(ly, M, W1, cur, 0, d);
                if (cm_has_local_begins) sc = fmaxf(sc, alpha[root_idx]);
                alpha[root_idx] = sc;
                /* Gate score: NULL3-corrected when requested (per-j outputs
                 * only; alpha stays raw). */
                float gate = sc;
                if (n3_act && sc >= n3.min_raw)
                    gate = sc - null3_correction(n3_act + (jp_g % W1_max) * 4,
                                                 n3_act + ((jp_g - d) % W1_max) * 4, d, n3);
                local_max = gpu_max_f(local_max, gate);
                if (d_dmin_j != NULL && gate >= dmin_cut) atomicMin(&s_dmin, d);
                /* Calibration: export root cells above a raw-score cut. */
                if (n3.cells != NULL && sc >= n3.cell_cut) {
                    int slot = atomicAdd(n3.cell_count, 1);
                    if (slot < n3.cell_room) {
                        int *cell = n3.cells + 4 * (int64_t)slot;
                        cell[0] = n3.cell_win_base + win;
                        cell[1] = (int)(j - i0);
                        cell[2] = d;
                        cell[3] = __float_as_int(sc);
                    }
                }
            }
            __syncthreads();
#elif defined(ROOT_D_OWNER)
            /* A fixed thread owns each root length for the entire local-begin
             * reduction. Preserve yy order while accumulating in a register;
             * no other thread accesses that root cell until the final barrier.
             * The legacy yy-outer loop changes ownership as dn_both changes,
             * so deleting its barriers without this reorganization is unsafe. */
            float local_max = GPU_IMPOSSIBLE;
            for (int d = dn0 + tid; d <= dx0; d += nthreads) {
                float sc = GPU_IMPOSSIBLE;
                sc = dp_combine<MAXPLUS>(sc, a_load<LY>(alpha, ly, M, W1, cur, y0, d) + tsc_0[0]);
                for (int yoff = 1; yoff < cn0; yoff++) {
                    sc = dp_combine<MAXPLUS>(sc, a_load<LY>(alpha, ly, M, W1, cur, y0 + yoff, d) + tsc_0[yoff]);
                }
                if (cm_has_local_begins) {
                    for (int yy = 1; yy < M; yy++) {
                        float begin = d_beginsc[yy];
                        if (!GPU_NOT_IMPOSSIBLE(begin) || d < d_dmin1[yy] || d > d_dmax1[yy]) continue;
                        float child;
                        if (d_stid[yy] == GPU_BEGL_S) {
                            child = b_load<LY>(alpha_begl, ly, cm_n_begl, W1, begl_slot<LY>(ly, d_begl_map[yy], j, W1), d_begl_map[yy], d, begl_packed_width, d_begl_offsets, d_begl_lo, d_begl_width);
                        } else {
                            child = alpha[a_idx<LY>(ly, M, W1, cur, yy, d)];
                        }
                        sc = fmaxf(sc, child + begin);
                    }
                }
                alpha[a_idx<LY>(ly, M, W1, cur, 0, d)] = sc;
                local_max = gpu_max_f(local_max, sc);
            }
            __syncthreads();
#else
            for (int d = dn0 + tid; d <= dx0; d += nthreads) {
                float sc = GPU_IMPOSSIBLE;
                sc = dp_combine<MAXPLUS>(sc, a_load<LY>(alpha, ly, M, W1, cur, y0, d) + tsc_0[0]);
                for (int yoff = 1; yoff < cn0; yoff++) {
                    sc = dp_combine<MAXPLUS>(sc, a_load<LY>(alpha, ly, M, W1, cur, y0 + yoff, d) + tsc_0[yoff]);
                }
                alpha[a_idx<LY>(ly, M, W1, cur, 0, d)] = sc;
            }
            __syncthreads();

            /* Local begins: Infernal's FastFInsideScan uses max (not logsum) for
               local begin contributions to ROOT. See cm_dpsearch.c lines 2221-2248:
               the FLogsum is commented out; actual code uses if(candidate > current).
               This hybrid strategy (logsum for transitions, max for local begins) is
               intentional — it finds the best entry point rather than summing. */
            if (cm_has_local_begins) {
                for (int yy = 1; yy < M; yy++) {
                    if (!GPU_NOT_IMPOSSIBLE(d_beginsc[yy])) continue;
                    int sid_yy  = d_stid[yy];
                    int dn_yy   = d_dmin1[yy];
                    int dx_yy   = d_dmax1[yy];
                    int dn_both = gpu_imax_ins(dn0, dn_yy);
                    int dx_both = gpu_imin_ins(dx0, dx_yy);

                    if (sid_yy == GPU_BEGL_S) {
                        int begl_yy = d_begl_map[yy];
                        int jp_yy   = begl_slot<LY>(ly, begl_yy, j, W1);
                        for (int d = dn_both + tid; d <= dx_both; d += nthreads) {
                            float candidate = b_load<LY>(alpha_begl, ly, cm_n_begl, W1, jp_yy, begl_yy, d, begl_packed_width, d_begl_offsets, d_begl_lo, d_begl_width) + d_beginsc[yy];
                            int64_t cell_idx = a_idx<LY>(ly, M, W1, cur, 0, d);
                            alpha[cell_idx] = fmaxf(alpha[cell_idx], candidate);
                        }
                    } else {
                        for (int d = dn_both + tid; d <= dx_both; d += nthreads) {
                            float candidate = alpha[a_idx<LY>(ly, M, W1, cur, yy, d)] + d_beginsc[yy];
                            int64_t cell_idx = a_idx<LY>(ly, M, W1, cur, 0, d);
                            alpha[cell_idx] = fmaxf(alpha[cell_idx], candidate);
                        }
                    }
                    __syncthreads();
                }
            }

            /* vsc_root: MAX over all d (finding best hit, not summing) */
            float local_max = GPU_IMPOSSIBLE;
            for (int d = dn0 + tid; d <= dx0; d += nthreads) {
                local_max = gpu_max_f(local_max, alpha[a_idx<LY>(ly, M, W1, cur, 0, d)]);
            }

#endif /* ROOT_D_OWNER */

            for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
                local_max = gpu_max_f(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
            }

            __shared__ float warp_maxes[32];
            int lane = tid % warpSize;
            int warp_id = tid / warpSize;
            if (lane == 0) warp_maxes[warp_id] = local_max;
            __syncthreads();

            if (tid == 0) {
                int n_warps = (nthreads + warpSize - 1) / warpSize;
                float block_max = GPU_IMPOSSIBLE;
                for (int w = 0; w < n_warps; w++) {
                    block_max = gpu_max_f(block_max, warp_maxes[w]);
                }
                vsc_root = gpu_max_f(vsc_root, block_max);
                /* bestsc[j] for cmcalibrate-style gamma hit resolution */
                if (d_bestsc_j != NULL) {
                    int jp0 = (int)(j - i0);   /* 0-based offset within window */
                    if (jp0 < perj_stride)
                        d_bestsc_j[(int64_t)win * perj_stride + jp0] = block_max;
                }
                if (d_dmin_j != NULL) {
                    int jp0 = (int)(j - i0);
                    if (jp0 < perj_stride)
                        d_dmin_j[(int64_t)win * perj_stride + jp0] = s_dmin;
                    s_dmin = 0x7fffffff;   /* next j's atomics follow several barriers */
                }
            }
            /* Optional: emit the full per-length root row alpha0[j][d] for d in
             * [dn0, dx0] (cmcalibrate's per-d bestsc[d]). All threads cooperate. */
            if (d_alpha0_jd != NULL) {
                int jp0 = (int)(j - i0);
                if (jp0 < perj_stride) {
                    int64_t row = ((int64_t)win * perj_stride + jp0) * alpha0_dstride;
                    for (int d = tid; d < alpha0_dstride; d += nthreads) {
                        float v = GPU_IMPOSSIBLE;
                        if (d >= dn0 && d <= dx0)
                            v = alpha[a_idx<LY>(ly, M, W1, cur, 0, d)];
                        d_alpha0_jd[row + d] = v;
                    }
                }
            }
            __syncthreads();
        }

    } /* end j-loop */

    if (tid == 0) {
        d_vsc_root[win] = vsc_root;
    }
}

__global__ void INSIDE_LAUNCH_BOUNDS
inside_scan_kernel(
    /* CM (read-only) */
    int         cm_M,
    int         cm_W,
    int         cm_abc_Kp,
    float       cm_el_selfsc,
    int         cm_has_local_begins,
    int         cm_n_begl,
    const int8_t  *__restrict__ d_sttype,
    const int8_t  *__restrict__ d_stid,
    const int     *__restrict__ d_cfirst,
    const int     *__restrict__ d_cnum,
    const int     *__restrict__ d_emitmode,
    const int     *__restrict__ d_sd,
    const int     *__restrict__ d_sdr,
    const float   *__restrict__ d_tsc,
    const float   *__restrict__ d_oesc,
    const int     *__restrict__ d_oesc_offset,
    const float   *__restrict__ d_beginsc,
    const float   *__restrict__ d_init_sc,
    const int     *__restrict__ d_begl_map,
    const int     *__restrict__ d_dmin1,
    const int     *__restrict__ d_dmax1,
    /* Batch windows */
    const uint8_t *__restrict__ d_dsq_batch,
    const int64_t *__restrict__ d_dsq_offset,
    const int64_t *__restrict__ d_win_i0,
    const int64_t *__restrict__ d_win_j0,
    int         n_windows,
    /* Workspace (pre-allocated per window) */
    float       *d_alpha,
    float       *d_alpha_begl,
    /* Output */
    float       *d_vsc_root,
    /* Optional per-end-position output: bestsc[j] = best local-Inside hit
     * score ENDING at position j (max over all lengths d and root local
     * begins). This is exactly the per-j input cmcalibrate feeds to its gamma
     * semi-HMM hit resolver. NULL => skip (legacy behavior). Stride = perj_stride
     * floats per window; bestsc[j] written at d_bestsc_j[win*perj_stride + (j-i0)]. */
    float       *d_bestsc_j,
    int          perj_stride,
    /* Optional FULL per-(j,d) root score output: alpha0[j][d] = best local-Inside
     * score of a hit of length d ending at j (this is cmcalibrate's per-length
     * bestsc[d], fed to UpdateGammaHitMx for ALL d). NULL => skip. Layout:
     * d_alpha0_jd[((int64)win*perj_stride + (j-i0))*alpha0_dstride + d]. */
    float       *d_alpha0_jd,
    int          alpha0_dstride,
    int          use_il_zero_emissions,
    int          begl_packed_width,
    const int *d_begl_offsets, const int *d_begl_lo, const int *d_begl_width,
    GPU_LevelSched sched,
    /* Optional per-j minimum root length d with score >= dmin_cut (NULL =>
     * skip; 0x7fffffff when none). Implemented for ROOT_STATE_FUSED builds;
     * other builds leave 0x7fffffff, which callers must treat as unknown. */
    float        dmin_cut,
    int         *d_dmin_j,
    GPU_Null3    n3,
    GPU_DPLayout ly
)
{
    dp_scan_body<false, 0>(cm_M, cm_W, cm_abc_Kp, cm_el_selfsc, cm_has_local_begins, cm_n_begl, d_sttype, d_stid, d_cfirst, d_cnum, d_emitmode, d_sd, d_sdr, d_tsc, d_oesc, d_oesc_offset, d_beginsc, d_init_sc, d_begl_map, d_dmin1, d_dmax1, d_dsq_batch, d_dsq_offset, d_win_i0, d_win_j0, n_windows, d_alpha, d_alpha_begl, d_vsc_root, d_bestsc_j, perj_stride, d_alpha0_jd, alpha0_dstride, use_il_zero_emissions, begl_packed_width, d_begl_offsets, d_begl_lo, d_begl_width, sched, dmin_cut, d_dmin_j, n3, ly);
}

/* CYK (max-plus) scan with the same schedule, storage packing and per-j/
 * per-(j,d) outputs as the Inside kernel. Infernal's CYK filter uses the
 * tighter beta1 bands; the caller uploads a device model with those bands. */
__global__ void INSIDE_LAUNCH_BOUNDS
cyk_fast_scan_kernel(
    /* CM (read-only) */
    int         cm_M,
    int         cm_W,
    int         cm_abc_Kp,
    float       cm_el_selfsc,
    int         cm_has_local_begins,
    int         cm_n_begl,
    const int8_t  *__restrict__ d_sttype,
    const int8_t  *__restrict__ d_stid,
    const int     *__restrict__ d_cfirst,
    const int     *__restrict__ d_cnum,
    const int     *__restrict__ d_emitmode,
    const int     *__restrict__ d_sd,
    const int     *__restrict__ d_sdr,
    const float   *__restrict__ d_tsc,
    const float   *__restrict__ d_oesc,
    const int     *__restrict__ d_oesc_offset,
    const float   *__restrict__ d_beginsc,
    const float   *__restrict__ d_init_sc,
    const int     *__restrict__ d_begl_map,
    const int     *__restrict__ d_dmin1,
    const int     *__restrict__ d_dmax1,
    /* Batch windows */
    const uint8_t *__restrict__ d_dsq_batch,
    const int64_t *__restrict__ d_dsq_offset,
    const int64_t *__restrict__ d_win_i0,
    const int64_t *__restrict__ d_win_j0,
    int         n_windows,
    /* Workspace (pre-allocated per window) */
    float       *d_alpha,
    float       *d_alpha_begl,
    /* Output */
    float       *d_vsc_root,
    /* Optional per-end-position output: bestsc[j] = best local-Inside hit
     * score ENDING at position j (max over all lengths d and root local
     * begins). This is exactly the per-j input cmcalibrate feeds to its gamma
     * semi-HMM hit resolver. NULL => skip (legacy behavior). Stride = perj_stride
     * floats per window; bestsc[j] written at d_bestsc_j[win*perj_stride + (j-i0)]. */
    float       *d_bestsc_j,
    int          perj_stride,
    /* Optional FULL per-(j,d) root score output: alpha0[j][d] = best local-Inside
     * score of a hit of length d ending at j (this is cmcalibrate's per-length
     * bestsc[d], fed to UpdateGammaHitMx for ALL d). NULL => skip. Layout:
     * d_alpha0_jd[((int64)win*perj_stride + (j-i0))*alpha0_dstride + d]. */
    float       *d_alpha0_jd,
    int          alpha0_dstride,
    int          use_il_zero_emissions,
    int          begl_packed_width,
    const int *d_begl_offsets, const int *d_begl_lo, const int *d_begl_width,
    GPU_LevelSched sched,
    /* Optional per-j minimum root length d with score >= dmin_cut (NULL =>
     * skip; 0x7fffffff when none). Implemented for ROOT_STATE_FUSED builds;
     * other builds leave 0x7fffffff, which callers must treat as unknown. */
    float        dmin_cut,
    int         *d_dmin_j,
    GPU_Null3    n3,
    GPU_DPLayout ly
)
{
    dp_scan_body<true, 0>(cm_M, cm_W, cm_abc_Kp, cm_el_selfsc, cm_has_local_begins, cm_n_begl, d_sttype, d_stid, d_cfirst, d_cnum, d_emitmode, d_sd, d_sdr, d_tsc, d_oesc, d_oesc_offset, d_beginsc, d_init_sc, d_begl_map, d_dmin1, d_dmax1, d_dsq_batch, d_dsq_offset, d_win_i0, d_win_j0, n_windows, d_alpha, d_alpha_begl, d_vsc_root, d_bestsc_j, perj_stride, d_alpha0_jd, alpha0_dstride, use_il_zero_emissions, begl_packed_width, d_begl_offsets, d_begl_lo, d_begl_width, sched, dmin_cut, d_dmin_j, n3, ly);
}


/* Fully packed layout instantiations (GPU_DPLayout). Built only with the
 * production switches; gpu_full_layout_supported() reports availability. */
#if defined(IL_PARALLEL_SCAN) && defined(ROOT_STATE_FUSED) && defined(BIF_RING_INDEX) && !defined(BIF_WARP_REDUCE) && !defined(ROOT_D_OWNER)
int gpu_full_layout_supported(void) { return 1; }
__global__ void INSIDE_LAUNCH_BOUNDS
inside_scan_kernel_full(
    /* CM (read-only) */
    int         cm_M,
    int         cm_W,
    int         cm_abc_Kp,
    float       cm_el_selfsc,
    int         cm_has_local_begins,
    int         cm_n_begl,
    const int8_t  *__restrict__ d_sttype,
    const int8_t  *__restrict__ d_stid,
    const int     *__restrict__ d_cfirst,
    const int     *__restrict__ d_cnum,
    const int     *__restrict__ d_emitmode,
    const int     *__restrict__ d_sd,
    const int     *__restrict__ d_sdr,
    const float   *__restrict__ d_tsc,
    const float   *__restrict__ d_oesc,
    const int     *__restrict__ d_oesc_offset,
    const float   *__restrict__ d_beginsc,
    const float   *__restrict__ d_init_sc,
    const int     *__restrict__ d_begl_map,
    const int     *__restrict__ d_dmin1,
    const int     *__restrict__ d_dmax1,
    /* Batch windows */
    const uint8_t *__restrict__ d_dsq_batch,
    const int64_t *__restrict__ d_dsq_offset,
    const int64_t *__restrict__ d_win_i0,
    const int64_t *__restrict__ d_win_j0,
    int         n_windows,
    /* Workspace (pre-allocated per window) */
    float       *d_alpha,
    float       *d_alpha_begl,
    /* Output */
    float       *d_vsc_root,
    /* Optional per-end-position output: bestsc[j] = best local-Inside hit
     * score ENDING at position j (max over all lengths d and root local
     * begins). This is exactly the per-j input cmcalibrate feeds to its gamma
     * semi-HMM hit resolver. NULL => skip (legacy behavior). Stride = perj_stride
     * floats per window; bestsc[j] written at d_bestsc_j[win*perj_stride + (j-i0)]. */
    float       *d_bestsc_j,
    int          perj_stride,
    /* Optional FULL per-(j,d) root score output: alpha0[j][d] = best local-Inside
     * score of a hit of length d ending at j (this is cmcalibrate's per-length
     * bestsc[d], fed to UpdateGammaHitMx for ALL d). NULL => skip. Layout:
     * d_alpha0_jd[((int64)win*perj_stride + (j-i0))*alpha0_dstride + d]. */
    float       *d_alpha0_jd,
    int          alpha0_dstride,
    int          use_il_zero_emissions,
    int          begl_packed_width,
    const int *d_begl_offsets, const int *d_begl_lo, const int *d_begl_width,
    GPU_LevelSched sched,
    /* Optional per-j minimum root length d with score >= dmin_cut (NULL =>
     * skip; 0x7fffffff when none). Implemented for ROOT_STATE_FUSED builds;
     * other builds leave 0x7fffffff, which callers must treat as unknown. */
    float        dmin_cut,
    int         *d_dmin_j,
    GPU_Null3    n3,
    GPU_DPLayout ly
)
{
    dp_scan_body<false, 1>(cm_M, cm_W, cm_abc_Kp, cm_el_selfsc, cm_has_local_begins, cm_n_begl, d_sttype, d_stid, d_cfirst, d_cnum, d_emitmode, d_sd, d_sdr, d_tsc, d_oesc, d_oesc_offset, d_beginsc, d_init_sc, d_begl_map, d_dmin1, d_dmax1, d_dsq_batch, d_dsq_offset, d_win_i0, d_win_j0, n_windows, d_alpha, d_alpha_begl, d_vsc_root, d_bestsc_j, perj_stride, d_alpha0_jd, alpha0_dstride, use_il_zero_emissions, begl_packed_width, d_begl_offsets, d_begl_lo, d_begl_width, sched, dmin_cut, d_dmin_j, n3, ly);
}

__global__ void INSIDE_LAUNCH_BOUNDS
cyk_fast_scan_kernel_full(
    /* CM (read-only) */
    int         cm_M,
    int         cm_W,
    int         cm_abc_Kp,
    float       cm_el_selfsc,
    int         cm_has_local_begins,
    int         cm_n_begl,
    const int8_t  *__restrict__ d_sttype,
    const int8_t  *__restrict__ d_stid,
    const int     *__restrict__ d_cfirst,
    const int     *__restrict__ d_cnum,
    const int     *__restrict__ d_emitmode,
    const int     *__restrict__ d_sd,
    const int     *__restrict__ d_sdr,
    const float   *__restrict__ d_tsc,
    const float   *__restrict__ d_oesc,
    const int     *__restrict__ d_oesc_offset,
    const float   *__restrict__ d_beginsc,
    const float   *__restrict__ d_init_sc,
    const int     *__restrict__ d_begl_map,
    const int     *__restrict__ d_dmin1,
    const int     *__restrict__ d_dmax1,
    /* Batch windows */
    const uint8_t *__restrict__ d_dsq_batch,
    const int64_t *__restrict__ d_dsq_offset,
    const int64_t *__restrict__ d_win_i0,
    const int64_t *__restrict__ d_win_j0,
    int         n_windows,
    /* Workspace (pre-allocated per window) */
    float       *d_alpha,
    float       *d_alpha_begl,
    /* Output */
    float       *d_vsc_root,
    /* Optional per-end-position output: bestsc[j] = best local-Inside hit
     * score ENDING at position j (max over all lengths d and root local
     * begins). This is exactly the per-j input cmcalibrate feeds to its gamma
     * semi-HMM hit resolver. NULL => skip (legacy behavior). Stride = perj_stride
     * floats per window; bestsc[j] written at d_bestsc_j[win*perj_stride + (j-i0)]. */
    float       *d_bestsc_j,
    int          perj_stride,
    /* Optional FULL per-(j,d) root score output: alpha0[j][d] = best local-Inside
     * score of a hit of length d ending at j (this is cmcalibrate's per-length
     * bestsc[d], fed to UpdateGammaHitMx for ALL d). NULL => skip. Layout:
     * d_alpha0_jd[((int64)win*perj_stride + (j-i0))*alpha0_dstride + d]. */
    float       *d_alpha0_jd,
    int          alpha0_dstride,
    int          use_il_zero_emissions,
    int          begl_packed_width,
    const int *d_begl_offsets, const int *d_begl_lo, const int *d_begl_width,
    GPU_LevelSched sched,
    /* Optional per-j minimum root length d with score >= dmin_cut (NULL =>
     * skip; 0x7fffffff when none). Implemented for ROOT_STATE_FUSED builds;
     * other builds leave 0x7fffffff, which callers must treat as unknown. */
    float        dmin_cut,
    int         *d_dmin_j,
    GPU_Null3    n3,
    GPU_DPLayout ly
)
{
    dp_scan_body<true, 1>(cm_M, cm_W, cm_abc_Kp, cm_el_selfsc, cm_has_local_begins, cm_n_begl, d_sttype, d_stid, d_cfirst, d_cnum, d_emitmode, d_sd, d_sdr, d_tsc, d_oesc, d_oesc_offset, d_beginsc, d_init_sc, d_begl_map, d_dmin1, d_dmax1, d_dsq_batch, d_dsq_offset, d_win_i0, d_win_j0, n_windows, d_alpha, d_alpha_begl, d_vsc_root, d_bestsc_j, perj_stride, d_alpha0_jd, alpha0_dstride, use_il_zero_emissions, begl_packed_width, d_begl_offsets, d_begl_lo, d_begl_width, sched, dmin_cut, d_dmin_j, n3, ly);
}

#else
/* Unsupported switch set: keep the symbols so hosts link; never launched
 * because gpu_full_layout_supported() returns 0. */
int gpu_full_layout_supported(void) { return 0; }
__global__ void INSIDE_LAUNCH_BOUNDS
inside_scan_kernel_full(
    /* CM (read-only) */
    int         cm_M,
    int         cm_W,
    int         cm_abc_Kp,
    float       cm_el_selfsc,
    int         cm_has_local_begins,
    int         cm_n_begl,
    const int8_t  *__restrict__ d_sttype,
    const int8_t  *__restrict__ d_stid,
    const int     *__restrict__ d_cfirst,
    const int     *__restrict__ d_cnum,
    const int     *__restrict__ d_emitmode,
    const int     *__restrict__ d_sd,
    const int     *__restrict__ d_sdr,
    const float   *__restrict__ d_tsc,
    const float   *__restrict__ d_oesc,
    const int     *__restrict__ d_oesc_offset,
    const float   *__restrict__ d_beginsc,
    const float   *__restrict__ d_init_sc,
    const int     *__restrict__ d_begl_map,
    const int     *__restrict__ d_dmin1,
    const int     *__restrict__ d_dmax1,
    /* Batch windows */
    const uint8_t *__restrict__ d_dsq_batch,
    const int64_t *__restrict__ d_dsq_offset,
    const int64_t *__restrict__ d_win_i0,
    const int64_t *__restrict__ d_win_j0,
    int         n_windows,
    /* Workspace (pre-allocated per window) */
    float       *d_alpha,
    float       *d_alpha_begl,
    /* Output */
    float       *d_vsc_root,
    /* Optional per-end-position output: bestsc[j] = best local-Inside hit
     * score ENDING at position j (max over all lengths d and root local
     * begins). This is exactly the per-j input cmcalibrate feeds to its gamma
     * semi-HMM hit resolver. NULL => skip (legacy behavior). Stride = perj_stride
     * floats per window; bestsc[j] written at d_bestsc_j[win*perj_stride + (j-i0)]. */
    float       *d_bestsc_j,
    int          perj_stride,
    /* Optional FULL per-(j,d) root score output: alpha0[j][d] = best local-Inside
     * score of a hit of length d ending at j (this is cmcalibrate's per-length
     * bestsc[d], fed to UpdateGammaHitMx for ALL d). NULL => skip. Layout:
     * d_alpha0_jd[((int64)win*perj_stride + (j-i0))*alpha0_dstride + d]. */
    float       *d_alpha0_jd,
    int          alpha0_dstride,
    int          use_il_zero_emissions,
    int          begl_packed_width,
    const int *d_begl_offsets, const int *d_begl_lo, const int *d_begl_width,
    GPU_LevelSched sched,
    /* Optional per-j minimum root length d with score >= dmin_cut (NULL =>
     * skip; 0x7fffffff when none). Implemented for ROOT_STATE_FUSED builds;
     * other builds leave 0x7fffffff, which callers must treat as unknown. */
    float        dmin_cut,
    int         *d_dmin_j,
    GPU_Null3    n3,
    GPU_DPLayout ly
)
{
    asm("trap;");
}

__global__ void INSIDE_LAUNCH_BOUNDS
cyk_fast_scan_kernel_full(
    /* CM (read-only) */
    int         cm_M,
    int         cm_W,
    int         cm_abc_Kp,
    float       cm_el_selfsc,
    int         cm_has_local_begins,
    int         cm_n_begl,
    const int8_t  *__restrict__ d_sttype,
    const int8_t  *__restrict__ d_stid,
    const int     *__restrict__ d_cfirst,
    const int     *__restrict__ d_cnum,
    const int     *__restrict__ d_emitmode,
    const int     *__restrict__ d_sd,
    const int     *__restrict__ d_sdr,
    const float   *__restrict__ d_tsc,
    const float   *__restrict__ d_oesc,
    const int     *__restrict__ d_oesc_offset,
    const float   *__restrict__ d_beginsc,
    const float   *__restrict__ d_init_sc,
    const int     *__restrict__ d_begl_map,
    const int     *__restrict__ d_dmin1,
    const int     *__restrict__ d_dmax1,
    /* Batch windows */
    const uint8_t *__restrict__ d_dsq_batch,
    const int64_t *__restrict__ d_dsq_offset,
    const int64_t *__restrict__ d_win_i0,
    const int64_t *__restrict__ d_win_j0,
    int         n_windows,
    /* Workspace (pre-allocated per window) */
    float       *d_alpha,
    float       *d_alpha_begl,
    /* Output */
    float       *d_vsc_root,
    /* Optional per-end-position output: bestsc[j] = best local-Inside hit
     * score ENDING at position j (max over all lengths d and root local
     * begins). This is exactly the per-j input cmcalibrate feeds to its gamma
     * semi-HMM hit resolver. NULL => skip (legacy behavior). Stride = perj_stride
     * floats per window; bestsc[j] written at d_bestsc_j[win*perj_stride + (j-i0)]. */
    float       *d_bestsc_j,
    int          perj_stride,
    /* Optional FULL per-(j,d) root score output: alpha0[j][d] = best local-Inside
     * score of a hit of length d ending at j (this is cmcalibrate's per-length
     * bestsc[d], fed to UpdateGammaHitMx for ALL d). NULL => skip. Layout:
     * d_alpha0_jd[((int64)win*perj_stride + (j-i0))*alpha0_dstride + d]. */
    float       *d_alpha0_jd,
    int          alpha0_dstride,
    int          use_il_zero_emissions,
    int          begl_packed_width,
    const int *d_begl_offsets, const int *d_begl_lo, const int *d_begl_width,
    GPU_LevelSched sched,
    /* Optional per-j minimum root length d with score >= dmin_cut (NULL =>
     * skip; 0x7fffffff when none). Implemented for ROOT_STATE_FUSED builds;
     * other builds leave 0x7fffffff, which callers must treat as unknown. */
    float        dmin_cut,
    int         *d_dmin_j,
    GPU_Null3    n3,
    GPU_DPLayout ly
)
{
    asm("trap;");
}

#endif
