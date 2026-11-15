/*
 * cyk_kernel.cu — CUDA kernel for CYK scanning DP on Covariance Models.
 *
 * This kernel mirrors Infernal 1.1.5's FastCYKScan() in cm_dpsearch.c.
 * Each CUDA block processes one sequence window independently.
 * Threads within a block cooperate on the d-dimension.
 *
 * Phase 2: QDB-banded mode — per-state dmin1[v]/dmax1[v] bands from
 * the CM file restrict the d-range, reducing compute and matching
 * Infernal's default search behavior.
 */

#include "gpu_types.h"

/* ────────────── device helper functions ────────────── */

__device__ __forceinline__
float gpu_max(float a, float b) {
    return fmaxf(a, b);
}

__host__ __device__ __forceinline__
int gpu_imax(int a, int b) { return a > b ? a : b; }

__host__ __device__ __forceinline__
int gpu_imin(int a, int b) { return a < b ? a : b; }

__host__ __device__ __forceinline__
int alpha_idx(int M, int W1, int j_row, int v, int d) {
    return j_row * (M * W1) + v * W1 + d;
}

__host__ __device__ __forceinline__
int alpha_begl_idx(int n_begl, int W1, int j_slot, int begl_idx, int d) {
    return j_slot * (n_begl * W1) + begl_idx * W1 + d;
}

/* ────────────── Main CYK Scan Kernel ────────────── */

__global__ void
cyk_scan_kernel(
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
    float       *d_vsc_root
)
{
    int win = blockIdx.x;
    if (win >= n_windows) return;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    int M  = cm_M;
    int Kp = cm_abc_Kp;

    int W_max = cm_W;
    int W1_max = W_max + 1;
    int alpha_win_stride      = 2 * M * W1_max;
    int alpha_begl_win_stride = W1_max * cm_n_begl * W1_max;
    float *alpha      = d_alpha      + (int64_t)win * alpha_win_stride;
    float *alpha_begl = d_alpha_begl + (int64_t)win * alpha_begl_win_stride;

    int64_t i0 = d_win_i0[win];
    int64_t j0 = d_win_j0[win];
    int64_t L  = j0 - i0 + 1;
    int W  = (W_max > (int)L) ? (int)L : W_max;

    const int W1 = W1_max;
    const uint8_t *dsq = d_dsq_batch + d_dsq_offset[win];

    /* Initialize alpha and alpha_begl to IMPOSSIBLE */
    for (int i = tid; i < alpha_win_stride; i += nthreads)
        alpha[i] = GPU_IMPOSSIBLE;
    for (int i = tid; i < alpha_begl_win_stride; i += nthreads)
        alpha_begl[i] = GPU_IMPOSSIBLE;
    __syncthreads();

    /* E_st base case */
    for (int v = tid; v < M; v += nthreads) {
        if (d_sttype[v] == GPU_E_st) {
            alpha[alpha_idx(M, W1_max, 0, v, 0)] = 0.0f;
            alpha[alpha_idx(M, W1_max, 1, v, 0)] = 0.0f;
        }
    }
    __syncthreads();

    float vsc_root = GPU_IMPOSSIBLE;

    for (int64_t j = i0; j <= j0; j++) {
        int cur  = (int)(j % 2);
        int prv  = (int)((j - 1) % 2);
        int jp_g = (int)(j - i0 + 1);

        for (int v = M - 1; v > 0; v--) {
            int st = d_sttype[v];
            if (st == GPU_E_st) continue;

            int sid       = d_stid[v];
            int em        = d_emitmode[v];
            int sd_v      = d_sd[v];
            int sdr_v     = d_sdr[v];
            int first     = d_cfirst[v];
            int cn        = d_cnum[v];
            int jp_v      = (sid == GPU_BEGL_S) ? ((int)(j % W1)) : cur;
            int jp_y      = (sdr_v > 0) ? prv : cur;

            /* QDB-banded d-range: [dmin1[v], min(dmax1[v], jp_g)] */
            int dn = d_dmin1[v];
            int dx = gpu_imin(d_dmax1[v], jp_g);

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

                for (int d = dn + tid; d <= dx; d += nthreads) {
                    /* k = d-value for right child y; d-k = d-value for left child w */
                    int kmin = gpu_imax(dmin1_y, d - dmax1_w);
                    int kmax = gpu_imin(dmax1_y, d - dmin1_w);
                    float sc = d_init_sc[v * W1 + d];
                    for (int k = kmin; k <= kmax; k++) {
                        int jp_wk = (int)((j - k) % W1);
                        if (jp_wk < 0) jp_wk += W1;
                        float left  = alpha_begl[alpha_begl_idx(cm_n_begl, W1, jp_wk, begl_w, d - k)];
                        float right = alpha[alpha_idx(M, W1, jp_y, y, k)];
                        sc = gpu_max(sc, left + right);
                    }
                    alpha[alpha_idx(M, W1, jp_v, v, d)] = sc;
                }
            }
            else if (sid == GPU_BEGL_S) {
                int begl_v = d_begl_map[v];
                int y = first;
                for (int d = dn + tid; d <= dx; d += nthreads) {
                    float sc = d_init_sc[v * W1 + d];
                    for (int yoff = 0; yoff < cn; yoff++) {
                        sc = gpu_max(sc, alpha[alpha_idx(M, W1, jp_y, y + yoff, d - sd_v)] + tsc_v[yoff]);
                    }
                    alpha_begl[alpha_begl_idx(cm_n_begl, W1, jp_v, begl_v, d)] = sc;
                }
            }
            else if (st == GPU_IL_st) {
                if (tid == 0) {
                    int y = first;
                    for (int d = dn; d <= dx; d++) {
                        int dp_y = d - sd_v;
                        float sc = d_init_sc[v * W1 + dp_y];
                        for (int yoff = 0; yoff < cn; yoff++)
                            sc = gpu_max(sc, alpha[alpha_idx(M, W1, jp_y, y + yoff, dp_y)] + tsc_v[yoff]);
                        int i = (int)(j - d + 1);
                        int off = d_oesc_offset[v];
                        sc += (off >= 0) ? d_oesc[off + dsq[i]] : 0.0f;
                        alpha[alpha_idx(M, W1, jp_v, v, d)] = sc;
                    }
                }
            }
            else if (st == GPU_IR_st) {
                int y = first;
                for (int d = dn + tid; d <= dx; d += nthreads) {
                    int dp_y = d - sd_v;
                    float sc = d_init_sc[v * W1 + dp_y];
                    for (int yoff = 0; yoff < cn; yoff++)
                        sc = gpu_max(sc, alpha[alpha_idx(M, W1, jp_y, y + yoff, dp_y)] + tsc_v[yoff]);
                    sc += esc_j;
                    alpha[alpha_idx(M, W1, jp_v, v, d)] = sc;
                }
            }
            else {
                int y = first;
                for (int d = dn + tid; d <= dx; d += nthreads) {
                    int dp_y = d - sd_v;
                    float sc = d_init_sc[v * W1 + dp_y];
                    for (int yoff = 0; yoff < cn; yoff++) {
                        sc = gpu_max(sc, alpha[alpha_idx(M, W1, jp_y, y + yoff, dp_y)] + tsc_v[yoff]);
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
                    alpha[alpha_idx(M, W1, jp_v, v, d)] = sc + esc;
                }
            }

            __syncthreads();
        } /* end v-loop */

        __syncthreads();

        /* ──── Handle ROOT (v=0) and local begins ──── */
        {
            const float *tsc_0 = d_tsc;
            int y0    = d_cfirst[0];
            int cn0   = d_cnum[0];
            int dn0   = d_dmin1[0];
            int dx0   = gpu_imin(d_dmax1[0], jp_g);

            for (int d = dn0 + tid; d <= dx0; d += nthreads) {
                float sc = GPU_IMPOSSIBLE;
                sc = gpu_max(sc, alpha[alpha_idx(M, W1, cur, y0, d)] + tsc_0[0]);
                for (int yoff = 1; yoff < cn0; yoff++) {
                    sc = gpu_max(sc, alpha[alpha_idx(M, W1, cur, y0 + yoff, d)] + tsc_0[yoff]);
                }
                alpha[alpha_idx(M, W1, cur, 0, d)] = sc;
            }
            __syncthreads();

            if (cm_has_local_begins) {
                for (int yy = 1; yy < M; yy++) {
                    if (!GPU_NOT_IMPOSSIBLE(d_beginsc[yy])) continue;
                    int sid_yy  = d_stid[yy];
                    int dn_yy   = d_dmin1[yy];
                    int dx_yy   = d_dmax1[yy];
                    int dn_both = gpu_imax(dn0, dn_yy);
                    int dx_both = gpu_imin(dx0, dx_yy);

                    if (sid_yy == GPU_BEGL_S) {
                        int begl_yy = d_begl_map[yy];
                        int jp_yy   = (int)(j % W1);
                        for (int d = dn_both + tid; d <= dx_both; d += nthreads) {
                            float candidate = alpha_begl[alpha_begl_idx(cm_n_begl, W1, jp_yy, begl_yy, d)] + d_beginsc[yy];
                            int cell_idx = alpha_idx(M, W1, cur, 0, d);
                            if (candidate > alpha[cell_idx])
                                alpha[cell_idx] = candidate;
                        }
                    } else {
                        for (int d = dn_both + tid; d <= dx_both; d += nthreads) {
                            float candidate = alpha[alpha_idx(M, W1, cur, yy, d)] + d_beginsc[yy];
                            int cell_idx = alpha_idx(M, W1, cur, 0, d);
                            if (candidate > alpha[cell_idx])
                                alpha[cell_idx] = candidate;
                        }
                    }
                    __syncthreads();
                }
            }

            float local_max = GPU_IMPOSSIBLE;
            for (int d = dn0 + tid; d <= dx0; d += nthreads) {
                local_max = gpu_max(local_max, alpha[alpha_idx(M, W1, cur, 0, d)]);
            }

            for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
                local_max = gpu_max(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
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
                    block_max = gpu_max(block_max, warp_maxes[w]);
                }
                vsc_root = gpu_max(vsc_root, block_max);
            }
            __syncthreads();
        }

    } /* end j-loop */

    if (tid == 0) {
        d_vsc_root[win] = vsc_root;
    }
}

