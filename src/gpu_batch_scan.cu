/*
 * gpu_batch_scan.cu — Host-side batch dispatch for GPU CYK scanning.
 *
 * Collects a batch of sequence windows, uploads them, launches the
 * CYK kernel, and retrieves per-window scores.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu_types.h"
#include "gpu_arena.h"

/* Declared in cyk_kernel.cu */
extern __global__ void
cyk_scan_kernel(
    int cm_M, int cm_W, int cm_abc_Kp, float cm_el_selfsc,
    int cm_has_local_begins, int cm_n_begl,
    const int8_t *d_sttype, const int8_t *d_stid,
    const int *d_cfirst, const int *d_cnum,
    const int *d_emitmode, const int *d_sd, const int *d_sdr,
    const float *d_tsc, const float *d_oesc, const int *d_oesc_offset,
    const float *d_beginsc, const float *d_init_sc, const int *d_begl_map,
    const int *d_dmin1, const int *d_dmax1,
    const uint8_t *d_dsq_batch, const int64_t *d_dsq_offset,
    const int64_t *d_win_i0, const int64_t *d_win_j0,
    int n_windows,
    float *d_alpha, float *d_alpha_begl,
    float *d_vsc_root
);

/* Declared in inside_kernel.cu */
extern __global__ void
inside_scan_kernel(
    int cm_M, int cm_W, int cm_abc_Kp, float cm_el_selfsc,
    int cm_has_local_begins, int cm_n_begl,
    const int8_t *d_sttype, const int8_t *d_stid,
    const int *d_cfirst, const int *d_cnum,
    const int *d_emitmode, const int *d_sd, const int *d_sdr,
    const float *d_tsc, const float *d_oesc, const int *d_oesc_offset,
    const float *d_beginsc, const float *d_init_sc, const int *d_begl_map,
    const int *d_dmin1, const int *d_dmax1,
    const uint8_t *d_dsq_batch, const int64_t *d_dsq_offset,
    const int64_t *d_win_i0, const int64_t *d_win_j0,
    int n_windows,
    float *d_alpha, float *d_alpha_begl,
    float *d_vsc_root,
    float *d_bestsc_j, int perj_stride,
    float *d_alpha0_jd, int alpha0_dstride,
    int use_il_zero_emissions,
    int begl_packed_width, const int *d_begl_offsets, const int *d_begl_lo, const int *d_begl_width,
    GPU_LevelSched sched,
    float dmin_cut, int *d_dmin_j, GPU_Null3 n3, GPU_DPLayout ly
);

/* Declared in inside_kernel.cu: max-plus instantiation of the same body */
extern __global__ void
cyk_fast_scan_kernel(
    int cm_M, int cm_W, int cm_abc_Kp, float cm_el_selfsc,
    int cm_has_local_begins, int cm_n_begl,
    const int8_t *d_sttype, const int8_t *d_stid,
    const int *d_cfirst, const int *d_cnum,
    const int *d_emitmode, const int *d_sd, const int *d_sdr,
    const float *d_tsc, const float *d_oesc, const int *d_oesc_offset,
    const float *d_beginsc, const float *d_init_sc, const int *d_begl_map,
    const int *d_dmin1, const int *d_dmax1,
    const uint8_t *d_dsq_batch, const int64_t *d_dsq_offset,
    const int64_t *d_win_i0, const int64_t *d_win_j0,
    int n_windows,
    float *d_alpha, float *d_alpha_begl,
    float *d_vsc_root,
    float *d_bestsc_j, int perj_stride,
    float *d_alpha0_jd, int alpha0_dstride,
    int use_il_zero_emissions,
    int begl_packed_width, const int *d_begl_offsets, const int *d_begl_lo, const int *d_begl_width,
    GPU_LevelSched sched,
    float dmin_cut, int *d_dmin_j, GPU_Null3 n3, GPU_DPLayout ly
);

/* Fully packed layout instantiations */
extern __global__ void
inside_scan_kernel_full(
    int cm_M, int cm_W, int cm_abc_Kp, float cm_el_selfsc,
    int cm_has_local_begins, int cm_n_begl,
    const int8_t *d_sttype, const int8_t *d_stid,
    const int *d_cfirst, const int *d_cnum,
    const int *d_emitmode, const int *d_sd, const int *d_sdr,
    const float *d_tsc, const float *d_oesc, const int *d_oesc_offset,
    const float *d_beginsc, const float *d_init_sc, const int *d_begl_map,
    const int *d_dmin1, const int *d_dmax1,
    const uint8_t *d_dsq_batch, const int64_t *d_dsq_offset,
    const int64_t *d_win_i0, const int64_t *d_win_j0,
    int n_windows,
    float *d_alpha, float *d_alpha_begl,
    float *d_vsc_root,
    float *d_bestsc_j, int perj_stride,
    float *d_alpha0_jd, int alpha0_dstride,
    int use_il_zero_emissions,
    int begl_packed_width, const int *d_begl_offsets, const int *d_begl_lo, const int *d_begl_width,
    GPU_LevelSched sched,
    float dmin_cut, int *d_dmin_j, GPU_Null3 n3, GPU_DPLayout ly
);

extern __global__ void
cyk_fast_scan_kernel_full(
    int cm_M, int cm_W, int cm_abc_Kp, float cm_el_selfsc,
    int cm_has_local_begins, int cm_n_begl,
    const int8_t *d_sttype, const int8_t *d_stid,
    const int *d_cfirst, const int *d_cnum,
    const int *d_emitmode, const int *d_sd, const int *d_sdr,
    const float *d_tsc, const float *d_oesc, const int *d_oesc_offset,
    const float *d_beginsc, const float *d_init_sc, const int *d_begl_map,
    const int *d_dmin1, const int *d_dmax1,
    const uint8_t *d_dsq_batch, const int64_t *d_dsq_offset,
    const int64_t *d_win_i0, const int64_t *d_win_j0,
    int n_windows,
    float *d_alpha, float *d_alpha_begl,
    float *d_vsc_root,
    float *d_bestsc_j, int perj_stride,
    float *d_alpha0_jd, int alpha0_dstride,
    int use_il_zero_emissions,
    int begl_packed_width, const int *d_begl_offsets, const int *d_begl_lo, const int *d_begl_width,
    GPU_LevelSched sched,
    float dmin_cut, int *d_dmin_j, GPU_Null3 n3, GPU_DPLayout ly
);

/* Kernel selection by workspace layout (identical signatures). */
typedef decltype(&inside_scan_kernel) dp_kernel_t;
static dp_kernel_t inside_kernel_for(const GPU_ScanWorkspace *ws) {
    return ws->full_layout ? inside_scan_kernel_full : inside_scan_kernel;
}
static dp_kernel_t cyk_kernel_for(const GPU_ScanWorkspace *ws) {
    return ws->full_layout ? cyk_fast_scan_kernel_full : cyk_fast_scan_kernel;
}
static GPU_DPLayout layout_of(const GPU_CM_Device *dev, const GPU_ScanWorkspace *ws) {
    GPU_DPLayout z = {};
    return ws->full_layout ? dev->full_layout : z;
}

/* Device NULL3 argument: copies the caller's configuration and allocates the
 * per-window composition ring (freed by release_null3). */
static int make_null3(const GPU_Null3 *cfg, int n_windows, int W, GPU_Null3 *out)
{
    *out = GPU_Null3{};
    if (!cfg || !cfg->enabled) return 0;
    *out = *cfg;
    out->act = NULL;
    out->cells = NULL; out->cell_count = NULL; out->cell_room = 0;
    if (gpu_dev_malloc(&out->act, (size_t)n_windows * (W + 1) * 4 * sizeof(float)) != cudaSuccess) return -1;
    return 0;
}

/* Root-cell emission (calibration): device buffer sized to the caller's
 * remaining room; cells_finish appends to the caller's buffer and adds the
 * number of qualifying cells to its count (above cell_cap = overflow). */
static int cells_begin(const GPU_Null3 *cfg, GPU_Null3 *n3)
{
    n3->cells = NULL; n3->cell_count = NULL; n3->cell_room = 0;
    if (!cfg || !cfg->h_cells || !cfg->h_cell_count) return 0;
    long room = (long)cfg->cell_cap - (long)*cfg->h_cell_count;
    if (room < 0) room = 0;
    if (gpu_dev_malloc(&n3->cells, (size_t)(room > 0 ? room : 1) * 4 * sizeof(int)) != cudaSuccess) return -1;
    if (gpu_dev_malloc(&n3->cell_count, sizeof(int)) != cudaSuccess) return -1;
    if (cudaMemset(n3->cell_count, 0, sizeof(int)) != cudaSuccess) return -1;
    n3->cell_room = (int)room;
    n3->cell_cut = cfg->cell_cut;
    n3->cell_win_base = cfg->cell_win_base;
    n3->cell_records = cfg->cell_records;
    n3->cell_margin = cfg->cell_margin;
    return 0;
}

static int cells_finish(const GPU_Null3 *cfg, GPU_Null3 *n3)
{
    if (!n3->cells) return 0;
    int got = 0;
    if (cudaMemcpy(&got, n3->cell_count, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    int keep = got < n3->cell_room ? got : n3->cell_room;
    if (keep > 0 && cudaMemcpy(cfg->h_cells + 4 * (size_t)*cfg->h_cell_count, n3->cells,
                               (size_t)keep * 4 * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    *cfg->h_cell_count += got;
    gpu_dev_free(n3->cells); gpu_dev_free(n3->cell_count);
    n3->cells = NULL; n3->cell_count = NULL;
    return 0;
}

/* Level schedule view for kernel launch; n_levels=0 means legacy loop. The
 * schedule is honored only by STATE_LEVEL_SCHED builds. */
static GPU_LevelSched level_sched_of(const GPU_CM_Device *dev)
{
    GPU_LevelSched s;
    s.n_levels          = dev->sched_n_levels;
    s.lvl_simple_begin  = dev->d_lvl_simple_begin;
    s.simple_state      = dev->d_simple_state;
    s.simple_item_start = dev->d_simple_item_start;
    s.lvl_il_begin      = dev->d_lvl_il_begin;
    s.il_state          = dev->d_il_state;
    return s;
}

#define CUDA_CHECK(call) do {                                             \
    cudaError_t err = (call);                                             \
    if (err != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                cudaGetErrorString(err));                                  \
        return -1;                                                        \
    }                                                                     \
} while(0)

/*
 * Tunable block size — default 128, override with GPU_BLOCK_SIZE env var.
 * Try 64/128/192/256 on different GPUs (A100 may prefer 256 for occupancy).
 */
static int
get_threads_per_block(void)
{
    const char *env = getenv("GPU_BLOCK_SIZE");
    if (env) {
        int v = atoi(env);
        if (v == 32 || v == 64 || v == 128 || v == 192 || v == 256)
            return v;
        fprintf(stderr, "[gpu_batch] invalid GPU_BLOCK_SIZE=%s, using 128\n", env);
    }
    return 128;
}

/* Safe for the public digital-sequence API as well as FASTA-based callers.
 * A model with a nonzero IL score or a batch containing gap/nonresidue codes
 * takes the original implementation. Sentinels are not sequence residues. */
static int il_zero_enabled(const GPU_CM_Device *dev, const GPU_Window *windows, int n_windows)
{
#ifdef IL_ZERO_EMISSION_SCAN
    if (!dev->il_nucleotide_emissions_zero || getenv("GPU_DISABLE_IL_ZERO")) return 0;
    for (int w = 0; w < n_windows; ++w) {
        for (int64_t j = windows[w].i0; j <= windows[w].j0; ++j) {
            uint8_t code = windows[w].dsq[j];
            if (code > 15 || code == 4) return 0;
        }
    }
    return 1;
#else
    return 0;
#endif
}

/* Blocks of inside_scan_kernel that can be resident at once on the current
 * device at the configured block size (occupancy x SM count). The scanner uses
 * it to size tiles so a model's tiles fill whole waves. */
int gpu_dp_resident_blocks(int cyk);

int
gpu_inside_resident_blocks(void)
{
    return gpu_dp_resident_blocks(0);
}

int
gpu_dp_resident_blocks(int cyk)
{
    /* Cached per process: occupancy depends only on kernel, device and block size. */
    static int cached[2] = { -1, -1 };
    if (cached[cyk ? 1 : 0] >= 0) return cached[cyk ? 1 : 0];
    int dev = 0, per_sm = 0;
    cudaDeviceProp prop;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
        return 0;
    cudaError_t rc = cyk
        ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, cyk_fast_scan_kernel, get_threads_per_block(), 0)
        : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, inside_scan_kernel, get_threads_per_block(), 0);
    if (rc != cudaSuccess)
        return 0;
    cached[cyk ? 1 : 0] = per_sm * prop.multiProcessorCount;
    return cached[cyk ? 1 : 0];
}

/* Internal: dispatch exactly n_windows (must be <= ws->max_batch) */
static int
gpu_batch_cyk_scan_single(
    const GPU_CM_Device     *dev,
    GPU_ScanWorkspace       *ws,
    const GPU_Window        *windows,
    int                      n_windows,
    float                   *scores
)
{
    if (n_windows <= 0) return 0;

    /* ── Step 1: Concatenate sequences and build offset/position arrays ── */
    int64_t total_residues = 0;
    for (int i = 0; i < n_windows; i++) {
        int64_t L = windows[i].j0 - windows[i].i0 + 1;
        total_residues += L + 2;
    }

    uint8_t *h_dsq_batch = (uint8_t *)calloc(total_residues, sizeof(uint8_t));
    int64_t *h_dsq_offset = (int64_t *)malloc(n_windows * sizeof(int64_t));
    int64_t *h_win_i0     = (int64_t *)malloc(n_windows * sizeof(int64_t));
    int64_t *h_win_j0     = (int64_t *)malloc(n_windows * sizeof(int64_t));

    int64_t off = 0;
    for (int i = 0; i < n_windows; i++) {
        int64_t L = windows[i].j0 - windows[i].i0 + 1;
        h_dsq_offset[i] = off;
        memcpy(h_dsq_batch + off, windows[i].dsq + windows[i].i0 - 1, (L + 2) * sizeof(uint8_t));
        h_win_i0[i] = 1;
        h_win_j0[i] = L;
        off += L + 2;
    }

    /* ── Step 2: Upload batch data to device ── */
    uint8_t *d_dsq_batch;
    int64_t *d_dsq_offset, *d_win_i0, *d_win_j0;

    CUDA_CHECK(gpu_dev_malloc(&d_dsq_batch,  total_residues * sizeof(uint8_t)));
    CUDA_CHECK(gpu_dev_malloc(&d_dsq_offset, n_windows * sizeof(int64_t)));
    CUDA_CHECK(gpu_dev_malloc(&d_win_i0,     n_windows * sizeof(int64_t)));
    CUDA_CHECK(gpu_dev_malloc(&d_win_j0,     n_windows * sizeof(int64_t)));

    CUDA_CHECK(cudaMemcpy(d_dsq_batch,  h_dsq_batch,  total_residues * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dsq_offset, h_dsq_offset, n_windows * sizeof(int64_t),      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_win_i0,     h_win_i0,     n_windows * sizeof(int64_t),      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_win_j0,     h_win_j0,     n_windows * sizeof(int64_t),      cudaMemcpyHostToDevice));

    /* ── Step 3: Launch kernel ── */
    int threads_per_block = get_threads_per_block();
    int blocks = n_windows;

    cyk_scan_kernel<<<blocks, threads_per_block>>>(
        dev->M, dev->W, dev->abc_Kp, dev->el_selfsc,
        dev->has_local_begins, dev->n_begl,
        dev->d_sttype, dev->d_stid, dev->d_cfirst, dev->d_cnum,
        dev->d_emitmode, dev->d_sd, dev->d_sdr,
        dev->d_tsc, dev->d_oesc, dev->d_oesc_offset,
        dev->d_beginsc, dev->d_init_sc, dev->d_begl_map,
        dev->d_dmin1, dev->d_dmax1,
        d_dsq_batch, d_dsq_offset, d_win_i0, d_win_j0,
        n_windows,
        ws->d_alpha, ws->d_alpha_begl,
        ws->d_vsc_root
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* ── Step 4: Retrieve results ── */
    CUDA_CHECK(cudaMemcpy(scores, ws->d_vsc_root, n_windows * sizeof(float), cudaMemcpyDeviceToHost));

    /* ── Cleanup batch-specific device memory ── */
    gpu_dev_free(d_dsq_batch);
    gpu_dev_free(d_dsq_offset);
    gpu_dev_free(d_win_i0);
    gpu_dev_free(d_win_j0);
    free(h_dsq_batch);
    free(h_dsq_offset);
    free(h_win_i0);
    free(h_win_j0);

    return 0;
}

/*
 * gpu_batch_cyk_scan()
 *
 * Run CYK scan on a batch of windows using a single CM.
 * If n_windows > ws->max_batch, automatically splits into sub-batches.
 *
 * Inputs:
 *   dev       — CM already uploaded to device
 *   ws        — pre-allocated scan workspace
 *   windows   — array of window descriptors (host)
 *   n_windows — number of windows (any size; auto-split if > max_batch)
 *
 * Outputs:
 *   scores    — [n_windows] best CYK score per window (host, caller-allocated)
 *
 * Returns 0 on success, -1 on error.
 */
int
gpu_batch_cyk_scan(
    const GPU_CM_Device     *dev,
    GPU_ScanWorkspace       *ws,
    const GPU_Window        *windows,
    int                      n_windows,
    float                   *scores
)
{
    if (ws->begl_packed_width > 0 || ws->full_layout) {
        fprintf(stderr, "[gpu_batch] packed workspace is not supported by the legacy CYK kernel\n");
        return -1;
    }
    if (n_windows <= 0) return 0;

    int max_b = ws->max_batch;
    if (n_windows <= max_b) {
        return gpu_batch_cyk_scan_single(dev, ws, windows, n_windows, scores);
    }

    /* Auto-split into sub-batches */
    int n_batches = (n_windows + max_b - 1) / max_b;
    fprintf(stderr, "[gpu_batch] auto-splitting %d windows into %d sub-batches of <=%d\n",
            n_windows, n_batches, max_b);

    for (int b = 0; b < n_batches; b++) {
        int start = b * max_b;
        int count = (start + max_b <= n_windows) ? max_b : (n_windows - start);
        int rc = gpu_batch_cyk_scan_single(dev, ws, windows + start, count, scores + start);
        if (rc != 0) return rc;
    }
    return 0;
}


/* ===================================================================
 *  Inside Algorithm — batch GPU scan
 * =================================================================== */

static int
gpu_batch_inside_scan_single(
    const GPU_CM_Device     *dev,
    GPU_ScanWorkspace       *ws,
    const GPU_Window        *windows,
    int                      n_windows,
    float                   *scores
)
{
    if (n_windows <= 0) return 0;

    int64_t total_residues = 0;
    for (int i = 0; i < n_windows; i++) {
        int64_t L = windows[i].j0 - windows[i].i0 + 1;
        total_residues += L + 2;
    }

    uint8_t *h_dsq_batch = (uint8_t *)calloc(total_residues, sizeof(uint8_t));
    int64_t *h_dsq_offset = (int64_t *)malloc(n_windows * sizeof(int64_t));
    int64_t *h_win_i0     = (int64_t *)malloc(n_windows * sizeof(int64_t));
    int64_t *h_win_j0     = (int64_t *)malloc(n_windows * sizeof(int64_t));

    int64_t off = 0;
    for (int i = 0; i < n_windows; i++) {
        int64_t L = windows[i].j0 - windows[i].i0 + 1;
        h_dsq_offset[i] = off;
        memcpy(h_dsq_batch + off, windows[i].dsq + windows[i].i0 - 1, (L + 2) * sizeof(uint8_t));
        h_win_i0[i] = 1;
        h_win_j0[i] = L;
        off += L + 2;
    }

    uint8_t *d_dsq_batch;
    int64_t *d_dsq_offset, *d_win_i0, *d_win_j0;

    CUDA_CHECK(gpu_dev_malloc(&d_dsq_batch,  total_residues * sizeof(uint8_t)));
    CUDA_CHECK(gpu_dev_malloc(&d_dsq_offset, n_windows * sizeof(int64_t)));
    CUDA_CHECK(gpu_dev_malloc(&d_win_i0,     n_windows * sizeof(int64_t)));
    CUDA_CHECK(gpu_dev_malloc(&d_win_j0,     n_windows * sizeof(int64_t)));

    CUDA_CHECK(cudaMemcpy(d_dsq_batch,  h_dsq_batch,  total_residues * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dsq_offset, h_dsq_offset, n_windows * sizeof(int64_t),      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_win_i0,     h_win_i0,     n_windows * sizeof(int64_t),      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_win_j0,     h_win_j0,     n_windows * sizeof(int64_t),      cudaMemcpyHostToDevice));

    int threads_per_block = get_threads_per_block();
    int blocks = n_windows;

    inside_kernel_for(ws)<<<blocks, threads_per_block>>>(
        dev->M, dev->W, dev->abc_Kp, dev->el_selfsc,
        dev->has_local_begins, dev->n_begl,
        dev->d_sttype, dev->d_stid, dev->d_cfirst, dev->d_cnum,
        dev->d_emitmode, dev->d_sd, dev->d_sdr,
        dev->d_tsc, dev->d_oesc, dev->d_oesc_offset,
        dev->d_beginsc, dev->d_init_sc, dev->d_begl_map,
        dev->d_dmin1, dev->d_dmax1,
        d_dsq_batch, d_dsq_offset, d_win_i0, d_win_j0,
        n_windows,
        ws->d_alpha, ws->d_alpha_begl,
        ws->d_vsc_root,
        NULL, 0,
        NULL, 0,
        il_zero_enabled(dev, windows, n_windows),
        ws->begl_packed_width, dev->d_begl_offsets, dev->d_begl_lo, dev->d_begl_width,
        level_sched_of(dev), 0.0f, NULL, GPU_Null3{}, layout_of(dev, ws)
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(scores, ws->d_vsc_root, n_windows * sizeof(float), cudaMemcpyDeviceToHost));

    gpu_dev_free(d_dsq_batch);
    gpu_dev_free(d_dsq_offset);
    gpu_dev_free(d_win_i0);
    gpu_dev_free(d_win_j0);
    free(h_dsq_batch);
    free(h_dsq_offset);
    free(h_win_i0);
    free(h_win_j0);

    return 0;
}

int
gpu_batch_inside_scan(
    const GPU_CM_Device     *dev,
    GPU_ScanWorkspace       *ws,
    const GPU_Window        *windows,
    int                      n_windows,
    float                   *scores
)
{
    if (n_windows <= 0) return 0;

    int max_b = ws->max_batch;
    if (n_windows <= max_b) {
        return gpu_batch_inside_scan_single(dev, ws, windows, n_windows, scores);
    }

    int n_batches = (n_windows + max_b - 1) / max_b;
    fprintf(stderr, "[gpu_batch_inside] auto-splitting %d windows into %d sub-batches of <=%d\n",
            n_windows, n_batches, max_b);

    for (int b = 0; b < n_batches; b++) {
        int start = b * max_b;
        int count = (start + max_b <= n_windows) ? max_b : (n_windows - start);
        int rc = gpu_batch_inside_scan_single(dev, ws, windows + start, count, scores + start);
        if (rc != 0) return rc;
    }
    return 0;
}


/* ===================================================================
 *  Inside Algorithm — batch GPU scan WITH per-end-position output.
 *
 *  For cmcalibrate-style E-value calibration: in addition to the per-window
 *  max Inside score, return bestsc[j] for every end position j of every
 *  window (the exact per-j input to cmcalibrate's gamma semi-HMM hit
 *  resolver). All windows must have the SAME length perj_stride.
 *  `bestsc_j` must be caller-allocated, size n_windows*perj_stride floats.
 * =================================================================== */

static int
gpu_batch_inside_scan_perj_single(
    const GPU_CM_Device *dev, GPU_ScanWorkspace *ws,
    const GPU_Window *windows, int n_windows,
    float *scores, float *bestsc_j, int perj_stride, const GPU_Null3 *n3_cfg)
{
    if (n_windows <= 0) return 0;

    int64_t total_residues = 0;
    for (int i = 0; i < n_windows; i++)
        total_residues += (windows[i].j0 - windows[i].i0 + 1) + 2;

    uint8_t *h_dsq_batch  = (uint8_t *)calloc(total_residues, sizeof(uint8_t));
    int64_t *h_dsq_offset = (int64_t *)malloc(n_windows * sizeof(int64_t));
    int64_t *h_win_i0     = (int64_t *)malloc(n_windows * sizeof(int64_t));
    int64_t *h_win_j0     = (int64_t *)malloc(n_windows * sizeof(int64_t));

    int64_t off = 0;
    for (int i = 0; i < n_windows; i++) {
        int64_t L = windows[i].j0 - windows[i].i0 + 1;
        h_dsq_offset[i] = off;
        memcpy(h_dsq_batch + off, windows[i].dsq + windows[i].i0 - 1, (L + 2) * sizeof(uint8_t));
        h_win_i0[i] = 1; h_win_j0[i] = L; off += L + 2;
    }

    uint8_t *d_dsq_batch; int64_t *d_dsq_offset, *d_win_i0, *d_win_j0;
    float   *d_bestsc_j;
    CUDA_CHECK(gpu_dev_malloc(&d_dsq_batch,  total_residues * sizeof(uint8_t)));
    CUDA_CHECK(gpu_dev_malloc(&d_dsq_offset, n_windows * sizeof(int64_t)));
    CUDA_CHECK(gpu_dev_malloc(&d_win_i0,     n_windows * sizeof(int64_t)));
    CUDA_CHECK(gpu_dev_malloc(&d_win_j0,     n_windows * sizeof(int64_t)));
    CUDA_CHECK(gpu_dev_malloc(&d_bestsc_j,   (size_t)n_windows * perj_stride * sizeof(float)));
    GPU_Null3 n3;
    CUDA_CHECK(make_null3(n3_cfg, n_windows, dev->W, &n3) == 0 ? cudaSuccess : cudaErrorMemoryAllocation);
    CUDA_CHECK(cells_begin(n3_cfg, &n3) == 0 ? cudaSuccess : cudaErrorMemoryAllocation);

    CUDA_CHECK(cudaMemcpy(d_dsq_batch,  h_dsq_batch,  total_residues * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dsq_offset, h_dsq_offset, n_windows * sizeof(int64_t),      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_win_i0,     h_win_i0,     n_windows * sizeof(int64_t),      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_win_j0,     h_win_j0,     n_windows * sizeof(int64_t),      cudaMemcpyHostToDevice));

    int threads_per_block = get_threads_per_block();
    inside_kernel_for(ws)<<<n_windows, threads_per_block>>>(
        dev->M, dev->W, dev->abc_Kp, dev->el_selfsc,
        dev->has_local_begins, dev->n_begl,
        dev->d_sttype, dev->d_stid, dev->d_cfirst, dev->d_cnum,
        dev->d_emitmode, dev->d_sd, dev->d_sdr,
        dev->d_tsc, dev->d_oesc, dev->d_oesc_offset,
        dev->d_beginsc, dev->d_init_sc, dev->d_begl_map,
        dev->d_dmin1, dev->d_dmax1,
        d_dsq_batch, d_dsq_offset, d_win_i0, d_win_j0,
        n_windows,
        ws->d_alpha, ws->d_alpha_begl,
        ws->d_vsc_root,
        d_bestsc_j, perj_stride,
        NULL, 0, il_zero_enabled(dev, windows, n_windows),
        ws->begl_packed_width, dev->d_begl_offsets, dev->d_begl_lo, dev->d_begl_width,
        level_sched_of(dev), 0.0f, NULL, n3, layout_of(dev, ws));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(scores, ws->d_vsc_root, n_windows * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bestsc_j, d_bestsc_j,
                          (size_t)n_windows * perj_stride * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cells_finish(n3_cfg, &n3) == 0 ? cudaSuccess : cudaErrorUnknown);

    gpu_dev_free(d_dsq_batch); gpu_dev_free(d_dsq_offset); gpu_dev_free(d_win_i0);
    gpu_dev_free(d_win_j0); gpu_dev_free(d_bestsc_j);
    gpu_dev_free(n3.act);
    free(h_dsq_batch); free(h_dsq_offset); free(h_win_i0); free(h_win_j0);
    return 0;
}

int
gpu_batch_inside_scan_perj(
    const GPU_CM_Device *dev, GPU_ScanWorkspace *ws,
    const GPU_Window *windows, int n_windows,
    float *scores, float *bestsc_j, int perj_stride, const GPU_Null3 *n3)
{
    if (n_windows <= 0) return 0;
    int max_b = ws->max_batch;
    if (n_windows <= max_b)
        return gpu_batch_inside_scan_perj_single(dev, ws, windows, n_windows,
                                                 scores, bestsc_j, perj_stride, n3);
    int n_batches = (n_windows + max_b - 1) / max_b;
    fprintf(stderr, "[gpu_batch_inside_perj] auto-splitting %d windows into %d sub-batches of <=%d\n",
            n_windows, n_batches, max_b);
    for (int b = 0; b < n_batches; b++) {
        int start = b * max_b;
        int count = (start + max_b <= n_windows) ? max_b : (n_windows - start);
        GPU_Null3 sub; const GPU_Null3 *subp = n3;
        if (n3) { sub = *n3; sub.cell_win_base = n3->cell_win_base + start; subp = &sub; }
        int rc = gpu_batch_inside_scan_perj_single(
            dev, ws, windows + start, count,
            scores + start, bestsc_j + (size_t)start * perj_stride, perj_stride, subp);
        if (rc != 0) return rc;
    }
    return 0;
}


/* ===================================================================
 *  Inside scan emitting the FULL per-(j,d) root score matrix alpha0[j][d]
 *  (cmcalibrate's per-length bestsc[d] for ALL d), so the exact gamma
 *  semi-HMM hit resolver can run on the host. All windows MUST be the same
 *  length == perj_stride. alpha0_jd is caller-allocated:
 *      n_windows * perj_stride * alpha0_dstride  floats,
 *  indexed [(win*perj_stride + (j-1))*alpha0_dstride + d].
 *  Memory-heavy: process in modest batches (caller controls n_windows).
 * =================================================================== */
static int
gpu_batch_inside_scan_alpha0_single(
    const GPU_CM_Device *dev, GPU_ScanWorkspace *ws,
    const GPU_Window *windows, int n_windows,
    float *scores, float *alpha0_jd, int perj_stride, int alpha0_dstride)
{
    if (n_windows <= 0) return 0;
    int64_t total_residues = 0;
    for (int i = 0; i < n_windows; i++)
        total_residues += (windows[i].j0 - windows[i].i0 + 1) + 2;

    uint8_t *h_dsq_batch  = (uint8_t *)calloc(total_residues, sizeof(uint8_t));
    int64_t *h_dsq_offset = (int64_t *)malloc(n_windows * sizeof(int64_t));
    int64_t *h_win_i0     = (int64_t *)malloc(n_windows * sizeof(int64_t));
    int64_t *h_win_j0     = (int64_t *)malloc(n_windows * sizeof(int64_t));
    int64_t off = 0;
    for (int i = 0; i < n_windows; i++) {
        int64_t L = windows[i].j0 - windows[i].i0 + 1;
        h_dsq_offset[i] = off;
        memcpy(h_dsq_batch + off, windows[i].dsq + windows[i].i0 - 1, (L + 2) * sizeof(uint8_t));
        h_win_i0[i] = 1; h_win_j0[i] = L; off += L + 2;
    }

    uint8_t *d_dsq_batch; int64_t *d_dsq_offset, *d_win_i0, *d_win_j0;
    float   *d_alpha0_jd;
    size_t  a0_count = (size_t)n_windows * perj_stride * alpha0_dstride;
    CUDA_CHECK(gpu_dev_malloc(&d_dsq_batch,  total_residues * sizeof(uint8_t)));
    CUDA_CHECK(gpu_dev_malloc(&d_dsq_offset, n_windows * sizeof(int64_t)));
    CUDA_CHECK(gpu_dev_malloc(&d_win_i0,     n_windows * sizeof(int64_t)));
    CUDA_CHECK(gpu_dev_malloc(&d_win_j0,     n_windows * sizeof(int64_t)));
    CUDA_CHECK(gpu_dev_malloc(&d_alpha0_jd,  a0_count * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_dsq_batch,  h_dsq_batch,  total_residues * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dsq_offset, h_dsq_offset, n_windows * sizeof(int64_t),      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_win_i0,     h_win_i0,     n_windows * sizeof(int64_t),      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_win_j0,     h_win_j0,     n_windows * sizeof(int64_t),      cudaMemcpyHostToDevice));

    int threads_per_block = get_threads_per_block();
    inside_kernel_for(ws)<<<n_windows, threads_per_block>>>(
        dev->M, dev->W, dev->abc_Kp, dev->el_selfsc,
        dev->has_local_begins, dev->n_begl,
        dev->d_sttype, dev->d_stid, dev->d_cfirst, dev->d_cnum,
        dev->d_emitmode, dev->d_sd, dev->d_sdr,
        dev->d_tsc, dev->d_oesc, dev->d_oesc_offset,
        dev->d_beginsc, dev->d_init_sc, dev->d_begl_map,
        dev->d_dmin1, dev->d_dmax1,
        d_dsq_batch, d_dsq_offset, d_win_i0, d_win_j0,
        n_windows,
        ws->d_alpha, ws->d_alpha_begl,
        ws->d_vsc_root,
        NULL, perj_stride,
        d_alpha0_jd, alpha0_dstride, il_zero_enabled(dev, windows, n_windows),
        ws->begl_packed_width, dev->d_begl_offsets, dev->d_begl_lo, dev->d_begl_width,
        level_sched_of(dev), 0.0f, NULL, GPU_Null3{}, layout_of(dev, ws));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(scores, ws->d_vsc_root, n_windows * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(alpha0_jd, d_alpha0_jd, a0_count * sizeof(float), cudaMemcpyDeviceToHost));

    gpu_dev_free(d_dsq_batch); gpu_dev_free(d_dsq_offset); gpu_dev_free(d_win_i0);
    gpu_dev_free(d_win_j0); gpu_dev_free(d_alpha0_jd);
    free(h_dsq_batch); free(h_dsq_offset); free(h_win_i0); free(h_win_j0);
    return 0;
}

int
gpu_batch_inside_scan_alpha0(
    const GPU_CM_Device *dev, GPU_ScanWorkspace *ws,
    const GPU_Window *windows, int n_windows,
    float *scores, float *alpha0_jd, int perj_stride, int alpha0_dstride)
{
    /* Caller is responsible for keeping n_windows small enough that the alpha0
     * buffer + workspace fit in GPU memory; we process exactly n_windows here. */
    return gpu_batch_inside_scan_alpha0_single(dev, ws, windows, n_windows,
                                               scores, alpha0_jd, perj_stride, alpha0_dstride);
}


/* ===================================================================
 *  E-value computation from Inside scores
 * =================================================================== */

float
gpu_evalue_from_score(float score, const GPU_EvalueParams *evp,
                      int64_t db_residues, int n_models)
{
    if (!evp->is_valid) return -1.0f;
    if (score < evp->mu) return (float)db_residues;

    /*
     * Infernal E-value formula (cmscan.c + stats.c + cm_tophits.c):
     *   pvalue = exp(-lambda * (score - mu_extrap))        [esl_exp_surv]
     *   Z      = n_models * db_residues * 2  (both strands)
     *   eZ     = (Z / calibration_db) * nrandhits          [cur_eff_dbsize]
     *   E      = pvalue * eZ
     *
     * Infernal multiplies Z by 2 for
     * the two strands it searches (`cfg->Z *= 2`, cmsearch.c:2458 / cmscan.c:592).
     * The previous code OMITTED this ×2, making E-values 2x too small (optimistic).
     * The old comment "×2 already in calibration params" was WRONG. `db_residues`
     * is the single-strand database size; we scan both strands, so multiply by 2.
     */
    double pvalue = exp(-(double)evp->lambda * ((double)score - (double)evp->mu));
    double Z = (double)n_models * (double)db_residues * 2.0;
    double eZ = (Z / evp->dbsize) * (double)evp->nrandhits;
    return (float)(pvalue * eZ);
}


/* ===================================================================
 *  Fast CYK (max-plus instantiation of the Inside kernel body).
 *  Same workspace, packing, schedule and output conventions as the Inside
 *  perj/alpha0 entry points. The device model must carry the bands intended
 *  for CYK (Infernal's CYK filter uses beta1 = SMX_QDB1_TIGHT).
 * =================================================================== */
static int
gpu_batch_cyk_fast_single(const GPU_CM_Device *dev, GPU_ScanWorkspace *ws,
                          const GPU_Window *windows, int n_windows, float *scores,
                          float *bestsc_j, int perj_stride, float *alpha0_jd, int alpha0_dstride,
                          float dmin_cut, int *dmin_j, const GPU_Null3 *n3_cfg)
{
    if (dmin_j && !bestsc_j) return -1;
    if (n_windows <= 0) return 0;
    int64_t total_residues = 0;
    for (int i = 0; i < n_windows; i++)
        total_residues += (windows[i].j0 - windows[i].i0 + 1) + 2;
    uint8_t *h_dsq_batch  = (uint8_t *)calloc(total_residues, sizeof(uint8_t));
    int64_t *h_dsq_offset = (int64_t *)malloc(n_windows * sizeof(int64_t));
    int64_t *h_win_i0     = (int64_t *)malloc(n_windows * sizeof(int64_t));
    int64_t *h_win_j0     = (int64_t *)malloc(n_windows * sizeof(int64_t));
    int64_t off = 0;
    for (int i = 0; i < n_windows; i++) {
        int64_t L = windows[i].j0 - windows[i].i0 + 1;
        h_dsq_offset[i] = off;
        memcpy(h_dsq_batch + off, windows[i].dsq + windows[i].i0 - 1, (L + 2) * sizeof(uint8_t));
        h_win_i0[i] = 1; h_win_j0[i] = L; off += L + 2;
    }
    uint8_t *d_dsq_batch; int64_t *d_dsq_offset, *d_win_i0, *d_win_j0;
    float *d_bestsc_j = NULL, *d_alpha0_jd = NULL;
    int *d_dmin_j = NULL;
    size_t perj_count = bestsc_j ? (size_t)n_windows * perj_stride : 0;
    size_t a0_count = alpha0_jd ? (size_t)n_windows * perj_stride * alpha0_dstride : 0;
    CUDA_CHECK(gpu_dev_malloc(&d_dsq_batch,  total_residues * sizeof(uint8_t)));
    CUDA_CHECK(gpu_dev_malloc(&d_dsq_offset, n_windows * sizeof(int64_t)));
    CUDA_CHECK(gpu_dev_malloc(&d_win_i0,     n_windows * sizeof(int64_t)));
    CUDA_CHECK(gpu_dev_malloc(&d_win_j0,     n_windows * sizeof(int64_t)));
    if (perj_count) CUDA_CHECK(gpu_dev_malloc(&d_bestsc_j, perj_count * sizeof(float)));
    if (a0_count)   CUDA_CHECK(gpu_dev_malloc(&d_alpha0_jd, a0_count * sizeof(float)));
    if (dmin_j)     CUDA_CHECK(gpu_dev_malloc(&d_dmin_j, perj_count * sizeof(int)));
    GPU_Null3 n3;
    CUDA_CHECK(make_null3(bestsc_j ? n3_cfg : NULL, n_windows, dev->W, &n3) == 0 ? cudaSuccess : cudaErrorMemoryAllocation);
    CUDA_CHECK(cells_begin(n3_cfg, &n3) == 0 ? cudaSuccess : cudaErrorMemoryAllocation);
    CUDA_CHECK(cudaMemcpy(d_dsq_batch,  h_dsq_batch,  total_residues * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dsq_offset, h_dsq_offset, n_windows * sizeof(int64_t),      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_win_i0,     h_win_i0,     n_windows * sizeof(int64_t),      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_win_j0,     h_win_j0,     n_windows * sizeof(int64_t),      cudaMemcpyHostToDevice));

    cyk_kernel_for(ws)<<<n_windows, get_threads_per_block()>>>(
        dev->M, dev->W, dev->abc_Kp, dev->el_selfsc,
        dev->has_local_begins, dev->n_begl,
        dev->d_sttype, dev->d_stid, dev->d_cfirst, dev->d_cnum,
        dev->d_emitmode, dev->d_sd, dev->d_sdr,
        dev->d_tsc, dev->d_oesc, dev->d_oesc_offset,
        dev->d_beginsc, dev->d_init_sc, dev->d_begl_map,
        dev->d_dmin1, dev->d_dmax1,
        d_dsq_batch, d_dsq_offset, d_win_i0, d_win_j0,
        n_windows,
        ws->d_alpha, ws->d_alpha_begl,
        ws->d_vsc_root,
        d_bestsc_j, perj_stride,
        d_alpha0_jd, alpha0_dstride, il_zero_enabled(dev, windows, n_windows),
        ws->begl_packed_width, dev->d_begl_offsets, dev->d_begl_lo, dev->d_begl_width,
        level_sched_of(dev), dmin_cut, d_dmin_j, n3, layout_of(dev, ws));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(scores, ws->d_vsc_root, n_windows * sizeof(float), cudaMemcpyDeviceToHost));
    if (dmin_j) CUDA_CHECK(cudaMemcpy(dmin_j, d_dmin_j, perj_count * sizeof(int), cudaMemcpyDeviceToHost));
    if (perj_count) CUDA_CHECK(cudaMemcpy(bestsc_j, d_bestsc_j, perj_count * sizeof(float), cudaMemcpyDeviceToHost));
    if (a0_count)   CUDA_CHECK(cudaMemcpy(alpha0_jd, d_alpha0_jd, a0_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cells_finish(n3_cfg, &n3) == 0 ? cudaSuccess : cudaErrorUnknown);
    gpu_dev_free(d_dsq_batch); gpu_dev_free(d_dsq_offset); gpu_dev_free(d_win_i0); gpu_dev_free(d_win_j0);
    gpu_dev_free(d_bestsc_j); gpu_dev_free(d_alpha0_jd); gpu_dev_free(d_dmin_j); gpu_dev_free(n3.act);
    free(h_dsq_batch); free(h_dsq_offset); free(h_win_i0); free(h_win_j0);
    return 0;
}

int
gpu_batch_cyk_fast_scan_perj(const GPU_CM_Device *dev, GPU_ScanWorkspace *ws,
                             const GPU_Window *windows, int n_windows,
                             float *scores, float *bestsc_j, int perj_stride,
                             float dmin_cut, int *dmin_j, const GPU_Null3 *n3)
{
    if (n_windows <= 0) return 0;
    int max_b = ws->max_batch;
    for (int start = 0; start < n_windows; start += max_b) {
        int count = (start + max_b <= n_windows) ? max_b : (n_windows - start);
        GPU_Null3 sub; const GPU_Null3 *subp = n3;
        if (n3) { sub = *n3; sub.cell_win_base = n3->cell_win_base + start; subp = &sub; }
        int rc = gpu_batch_cyk_fast_single(dev, ws, windows + start, count, scores + start,
                                           bestsc_j ? bestsc_j + (size_t)start * perj_stride : NULL,
                                           perj_stride, NULL, 0, dmin_cut,
                                           dmin_j ? dmin_j + (size_t)start * perj_stride : NULL, subp);
        if (rc != 0) return rc;
    }
    return 0;
}

int
gpu_batch_cyk_fast_scan_alpha0(const GPU_CM_Device *dev, GPU_ScanWorkspace *ws,
                               const GPU_Window *windows, int n_windows,
                               float *scores, float *alpha0_jd, int perj_stride, int alpha0_dstride)
{
    return gpu_batch_cyk_fast_single(dev, ws, windows, n_windows, scores, NULL, perj_stride,
                                     alpha0_jd, alpha0_dstride, 0.0f, NULL, NULL);
}
