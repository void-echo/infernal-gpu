/*
 * gpu_transfer.cu — Marshal Infernal CM data to/from GPU.
 *
 * Converts the host-side GPU_CM_Host (already flattened from CM_t)
 * into device-side GPU_CM_Device by allocating and copying to GPU memory.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu_types.h"
#include "gpu_arena.h"

#define CUDA_CHECK(call) do {                                             \
    cudaError_t err = (call);                                             \
    if (err != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                cudaGetErrorString(err));                                  \
        return -1;                                                        \
    }                                                                     \
} while(0)

/* ──── Allocate and populate a GPU_CM_Host from raw CM parameters ──── */

GPU_CM_Host *
gpu_cm_host_create(
    int M, int W, int abc_Kp, float el_selfsc, int has_local_begins,
    const int8_t *sttype, const int8_t *stid,
    const int *cfirst, const int *cnum,
    const float *tsc_flat,        /* [M * MAXCONNECT] */
    const float *oesc_flat,       /* [oesc_total] */
    const int *oesc_offset,       /* [M] */
    int oesc_total,
    const float *beginsc,         /* [M] */
    const float *endsc,           /* [M] */
    const float *init_sc_flat,    /* [M * (W+1)] */
    const int *dmin1,             /* [M] QDB bands; NULL → sd[v] */
    const int *dmax1              /* [M] QDB bands; NULL → W     */
)
{
    GPU_CM_Host *h = (GPU_CM_Host *)calloc(1, sizeof(GPU_CM_Host));
    /* calloc zeros everything including evp_* fields and name/acc */
    h->M = M;
    h->W = W;
    h->abc_Kp = abc_Kp;
    h->el_selfsc = el_selfsc;
    h->has_local_begins = has_local_begins;

    int W1 = W + 1;

    /* Copy per-state topology */
    h->sttype   = (int8_t *)malloc(M * sizeof(int8_t));
    h->stid     = (int8_t *)malloc(M * sizeof(int8_t));
    h->cfirst   = (int *)malloc(M * sizeof(int));
    h->cnum     = (int *)malloc(M * sizeof(int));
    h->emitmode = (int *)malloc(M * sizeof(int));
    h->sd       = (int *)malloc(M * sizeof(int));
    h->sdr      = (int *)malloc(M * sizeof(int));

    memcpy(h->sttype, sttype, M * sizeof(int8_t));
    memcpy(h->stid,   stid,   M * sizeof(int8_t));
    memcpy(h->cfirst, cfirst, M * sizeof(int));
    memcpy(h->cnum,   cnum,   M * sizeof(int));

    /* Precompute emitmode, sd, sdr from sttype */
    for (int v = 0; v < M; v++) {
        int st = sttype[v];
        /* Emitmode */
        switch (st) {
            case GPU_IL_st: case GPU_ML_st: h->emitmode[v] = GPU_EMITLEFT;  break;
            case GPU_IR_st: case GPU_MR_st: h->emitmode[v] = GPU_EMITRIGHT; break;
            case GPU_MP_st:                 h->emitmode[v] = GPU_EMITPAIR;  break;
            default:                        h->emitmode[v] = GPU_EMITNONE;  break;
        }
        /* StateDelta */
        switch (st) {
            case GPU_MP_st: h->sd[v] = 2; break;
            case GPU_ML_st: case GPU_MR_st: case GPU_IL_st: case GPU_IR_st: h->sd[v] = 1; break;
            default: h->sd[v] = 0; break;
        }
        /* StateRightDelta */
        switch (st) {
            case GPU_MP_st: case GPU_MR_st: case GPU_IR_st: h->sdr[v] = 1; break;
            default: h->sdr[v] = 0; break;
        }
    }

    /* Transition scores */
    h->tsc = (float *)malloc((size_t)M * GPU_MAXCONNECT * sizeof(float));
    memcpy(h->tsc, tsc_flat, (size_t)M * GPU_MAXCONNECT * sizeof(float));

    /* Emission scores */
    h->oesc_total = oesc_total;
    h->oesc = (float *)malloc((size_t)oesc_total * sizeof(float));
    memcpy(h->oesc, oesc_flat, (size_t)oesc_total * sizeof(float));
    h->oesc_offset = (int *)malloc(M * sizeof(int));
    memcpy(h->oesc_offset, oesc_offset, M * sizeof(int));

    /* Local alignment scores */
    h->beginsc = (float *)malloc(M * sizeof(float));
    h->endsc   = (float *)malloc(M * sizeof(float));
    memcpy(h->beginsc, beginsc, M * sizeof(float));
    memcpy(h->endsc,   endsc,   M * sizeof(float));

    /* init_sc */
    h->init_sc = (float *)malloc((size_t)M * W1 * sizeof(float));
    memcpy(h->init_sc, init_sc_flat, (size_t)M * W1 * sizeof(float));

    /* QDB bands */
    h->dmin1 = (int *)malloc(M * sizeof(int));
    h->dmax1 = (int *)malloc(M * sizeof(int));
    if (dmin1 && dmax1) {
        memcpy(h->dmin1, dmin1, M * sizeof(int));
        memcpy(h->dmax1, dmax1, M * sizeof(int));
    } else {
        for (int v = 0; v < M; v++) {
            h->dmin1[v] = h->sd[v];
            h->dmax1[v] = W;
        }
    }

    /* Identify BEGL_S states */
    h->begl_map  = (int *)malloc(M * sizeof(int));
    int n_begl = 0;
    for (int v = 0; v < M; v++) {
        if (stid[v] == GPU_BEGL_S) {
            h->begl_map[v] = n_begl++;
        } else {
            h->begl_map[v] = -1;
        }
    }
    h->n_begl = n_begl;
    h->begl_list = (int *)malloc(n_begl * sizeof(int));
    int idx = 0;
    for (int v = 0; v < M; v++) {
        if (stid[v] == GPU_BEGL_S)
            h->begl_list[idx++] = v;
    }

    return h;
}

void
gpu_cm_host_destroy(GPU_CM_Host *h)
{
    if (!h) return;
    free(h->sttype);   free(h->stid);
    free(h->cfirst);   free(h->cnum);
    free(h->emitmode); free(h->sd); free(h->sdr);
    free(h->tsc);      free(h->oesc);  free(h->oesc_offset);
    free(h->beginsc);  free(h->endsc); free(h->init_sc);
    free(h->dmin1);    free(h->dmax1);
    free(h->begl_map); free(h->begl_list);
    free(h);
}

int gpu_build_level_schedule(const GPU_CM_Host *h, GPU_CM_Device *dev);
int gpu_build_full_layout(const GPU_CM_Host *h, GPU_CM_Device *dev);

/* ──── Upload GPU_CM_Host → GPU_CM_Device ──── */

int
gpu_cm_upload(const GPU_CM_Host *h, GPU_CM_Device *dev)
{
    int M  = h->M;
    int W1 = h->W + 1;

    dev->M = M;
    dev->W = h->W;
    dev->abc_Kp = h->abc_Kp;
    dev->el_selfsc = h->el_selfsc;
    dev->has_local_begins = h->has_local_begins;
    dev->n_begl = h->n_begl;

    /* Model-level proof for the optional zero-emission IL fast path.
     * Gap/nonresidue/missing are excluded here; the batch dispatcher must
     * independently reject such digital codes before enabling the path. */
    dev->il_nucleotide_emissions_zero = (h->abc_Kp >= 16);
    for (int v = 0; v < h->M && dev->il_nucleotide_emissions_zero; ++v) {
        if (h->sttype[v] != GPU_IL_st || h->oesc_offset[v] < 0) continue;
        for (int code = 0; code <= 15; ++code) {
            if (code == 4) continue;
            if (h->oesc[h->oesc_offset[v] + code] != 0.0f)
                dev->il_nucleotide_emissions_zero = 0;
        }
    }

    dev->begl_packed_width = 0;
    dev->d_begl_offsets = NULL; dev->d_begl_lo = NULL; dev->d_begl_width = NULL;
    if (h->n_begl > 0) {
        int *offsets = (int *)malloc(h->n_begl * sizeof(int));
        int *lo = (int *)malloc(h->n_begl * sizeof(int));
        int *width = (int *)malloc(h->n_begl * sizeof(int));
        for (int b = 0; b < h->n_begl; ++b) {
            int v = h->begl_list[b];
            offsets[b] = dev->begl_packed_width;
            lo[b] = h->dmin1[v];
            width[b] = h->dmax1[v] - lo[b] + 1;
            if (width[b] < 0) width[b] = 0;
            dev->begl_packed_width += width[b];
        }
        CUDA_CHECK(gpu_dev_malloc(&dev->d_begl_offsets, h->n_begl * sizeof(int)));
        CUDA_CHECK(gpu_dev_malloc(&dev->d_begl_lo, h->n_begl * sizeof(int)));
        CUDA_CHECK(gpu_dev_malloc(&dev->d_begl_width, h->n_begl * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(dev->d_begl_offsets, offsets, h->n_begl * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev->d_begl_lo, lo, h->n_begl * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev->d_begl_width, width, h->n_begl * sizeof(int), cudaMemcpyHostToDevice));
        free(offsets); free(lo); free(width);
    }

    /* State topology */
    CUDA_CHECK(gpu_dev_malloc(&dev->d_sttype,   M * sizeof(int8_t)));
    CUDA_CHECK(gpu_dev_malloc(&dev->d_stid,     M * sizeof(int8_t)));
    CUDA_CHECK(gpu_dev_malloc(&dev->d_cfirst,   M * sizeof(int)));
    CUDA_CHECK(gpu_dev_malloc(&dev->d_cnum,     M * sizeof(int)));
    CUDA_CHECK(gpu_dev_malloc(&dev->d_emitmode, M * sizeof(int)));
    CUDA_CHECK(gpu_dev_malloc(&dev->d_sd,       M * sizeof(int)));
    CUDA_CHECK(gpu_dev_malloc(&dev->d_sdr,      M * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dev->d_sttype,   h->sttype,   M * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev->d_stid,     h->stid,     M * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev->d_cfirst,   h->cfirst,   M * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev->d_cnum,     h->cnum,     M * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev->d_emitmode, h->emitmode, M * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev->d_sd,       h->sd,       M * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev->d_sdr,      h->sdr,      M * sizeof(int),    cudaMemcpyHostToDevice));

    /* Scores */
    size_t tsc_bytes = (size_t)M * GPU_MAXCONNECT * sizeof(float);
    CUDA_CHECK(gpu_dev_malloc(&dev->d_tsc, tsc_bytes));
    CUDA_CHECK(cudaMemcpy(dev->d_tsc, h->tsc, tsc_bytes, cudaMemcpyHostToDevice));

    size_t oesc_bytes = (size_t)h->oesc_total * sizeof(float);
    CUDA_CHECK(gpu_dev_malloc(&dev->d_oesc, oesc_bytes));
    CUDA_CHECK(cudaMemcpy(dev->d_oesc, h->oesc, oesc_bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(gpu_dev_malloc(&dev->d_oesc_offset, M * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dev->d_oesc_offset, h->oesc_offset, M * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(gpu_dev_malloc(&dev->d_beginsc, M * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dev->d_beginsc, h->beginsc, M * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(gpu_dev_malloc(&dev->d_endsc, M * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dev->d_endsc, h->endsc, M * sizeof(float), cudaMemcpyHostToDevice));

    size_t init_bytes = (size_t)M * W1 * sizeof(float);
    CUDA_CHECK(gpu_dev_malloc(&dev->d_init_sc, init_bytes));
    CUDA_CHECK(cudaMemcpy(dev->d_init_sc, h->init_sc, init_bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(gpu_dev_malloc(&dev->d_begl_map, M * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dev->d_begl_map, h->begl_map, M * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(gpu_dev_malloc(&dev->d_dmin1, M * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dev->d_dmin1, h->dmin1, M * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(gpu_dev_malloc(&dev->d_dmax1, M * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dev->d_dmax1, h->dmax1, M * sizeof(int), cudaMemcpyHostToDevice));

    if (gpu_build_full_layout(h, dev) != 0) return -1;

    dev->sched_n_levels = 0;
    dev->d_sched_buf = NULL;
    const char *sched_env = getenv("GPU_LEVEL_SCHED");
    if (!sched_env || strcmp(sched_env, "0") != 0)
        if (gpu_build_level_schedule(h, dev) != 0) return -1;

    return 0;
}

/* Fully packed layout: per-state alpha bands and per-BEGL ring lengths.
 * A BEGL state is read only by its parent B (slot j-k, k <= min(dmax(BEGR),W))
 * and by root local begins (k = 0), so min(dmax(BEGR sibling), W) + 1 history
 * rows suffice. */
int
gpu_build_full_layout(const GPU_CM_Host *h, GPU_CM_Device *dev)
{
    int M = h->M, nb = h->n_begl;
    size_t n_ints = (size_t)3 * M + 2 * (size_t)(nb > 0 ? nb : 1);
    int *buf = (int *)calloc(n_ints, sizeof(int));
    int *off = buf, *lo = off + M, *hi = lo + M, *base = hi + M, *ring = base + (nb > 0 ? nb : 1);
    long long width = 0;
    for (int v = 0; v < M; v++) {
        if (h->sttype[v] == GPU_E_st) { lo[v] = 0; hi[v] = 0; }
        else { lo[v] = h->dmin1[v]; hi[v] = h->dmax1[v] < h->W ? h->dmax1[v] : h->W; }
        off[v] = (int)width;
        if (hi[v] >= lo[v]) width += hi[v] - lo[v] + 1;
    }
    for (int b = 0; b < nb; b++) ring[b] = 1;
    for (int v = 0; v < M; v++) {
        if (h->sttype[v] != GPU_B_st) continue;
        int w = h->cfirst[v], y = h->cnum[v], b = h->begl_map[w];
        if (b < 0) { free(buf); fprintf(stderr, "[full_layout] B state without BEGL child\n"); return -1; }
        int need = (h->dmax1[y] < h->W ? h->dmax1[y] : h->W) + 1;
        if (need > ring[b]) ring[b] = need;
    }
    long long begl_floats = 0;
    for (int b = 0; b < nb; b++) {
        int v = h->begl_list[b];
        int bw = h->dmax1[v] - h->dmin1[v] + 1;
        base[b] = (int)begl_floats;
        begl_floats += (long long)ring[b] * (bw > 0 ? bw : 0);
    }
    if (width > INT32_MAX || begl_floats > INT32_MAX) { free(buf); fprintf(stderr, "[full_layout] layout too large\n"); return -1; }
    if (gpu_dev_malloc(&dev->d_layout_buf, n_ints * sizeof(int)) != cudaSuccess ||
        cudaMemcpy(dev->d_layout_buf, buf, n_ints * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        free(buf); fprintf(stderr, "[full_layout] device allocation failed\n"); return -1;
    }
    free(buf);
    GPU_DPLayout L;
    L.full = 1;
    L.a_width = (int)width;
    L.begl_floats = begl_floats;
    L.a_off = dev->d_layout_buf; L.a_lo = L.a_off + M; L.a_hi = L.a_lo + M;
    L.b_base = L.a_hi + M; L.b_ring = L.b_base + (nb > 0 ? nb : 1);
    dev->full_layout = L;
    return 0;
}

/* Dependency levels for one sequence position j. Children indices exceed the
 * parent (except insert self loops), so one descending pass suffices.
 *   sdr>0 (MP/MR/IR): children come from row j-1 -> level 0.
 *   otherwise: 1 + max level of same-row children (E states hold constants).
 * B reads its BEGL child (ring slot j-k, including k=0) and BEGR child.
 * IL self loops are resolved inside the IL prefix scan, so the self edge is
 * not a scheduling dependency. Every cell keeps its legacy operation order. */
int
gpu_build_level_schedule(const GPU_CM_Host *h, GPU_CM_Device *dev)
{
    int M = h->M;
    int *level = (int *)malloc(M * sizeof(int));
    int n_levels = 0;
    for (int v = M - 1; v > 0; v--) {
        level[v] = -1;
        int st = h->sttype[v];
        if (st == GPU_E_st) continue;
        int lv = 0;
        if (h->sdr[v] == 0) {
            if (st == GPU_B_st) {
                int kids[2] = { h->cfirst[v], h->cnum[v] };
                for (int q = 0; q < 2; q++)
                    if (level[kids[q]] >= 0 && level[kids[q]] + 1 > lv) lv = level[kids[q]] + 1;
            } else {
                for (int y = h->cfirst[v]; y < h->cfirst[v] + h->cnum[v]; y++) {
                    if (y == v || h->sttype[y] == GPU_E_st) continue;
                    if (y < v) { free(level); fprintf(stderr, "[level_sched] unexpected child order\n"); return -1; }
                    if (level[y] + 1 > lv) lv = level[y] + 1;
                }
            }
        }
        level[v] = lv;
        if (lv + 1 > n_levels) n_levels = lv + 1;
    }
    int n_simple = 0, n_il = 0;
    for (int v = 1; v < M; v++) {
        if (level[v] < 0) continue;
        if (h->sttype[v] == GPU_IL_st) n_il++; else n_simple++;
    }
    /* layout: lvl_simple_begin[L+1] simple_state[S] simple_item_start[S+1]
     *         lvl_il_begin[L+1] il_state[I] */
    size_t n_ints = (size_t)(n_levels + 1) * 2 + n_simple * 2 + 1 + n_il;
    int *buf = (int *)calloc(n_ints, sizeof(int));
    int *lsb = buf, *ss = lsb + n_levels + 1, *sis = ss + n_simple;
    int *lib = sis + n_simple + 1, *ils = lib + n_levels + 1;
    int ks = 0, ki = 0, items = 0;
    for (int l = 0; l < n_levels; l++) {
        lsb[l] = ks; lib[l] = ki;
        for (int v = M - 1; v > 0; v--) {   /* descending v within a level */
            if (level[v] != l) continue;
            if (h->sttype[v] == GPU_IL_st) { ils[ki++] = v; continue; }
            ss[ks] = v; sis[ks] = items; ks++;
            int width = h->dmax1[v] - h->dmin1[v] + 1;
            items += width > 0 ? width : 0;
        }
    }
    lsb[n_levels] = ks; lib[n_levels] = ki; sis[n_simple] = items;
    free(level);
    if (gpu_dev_malloc(&dev->d_sched_buf, n_ints * sizeof(int)) != cudaSuccess ||
        cudaMemcpy(dev->d_sched_buf, buf, n_ints * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        free(buf);
        fprintf(stderr, "[level_sched] device allocation failed\n");
        return -1;
    }
    free(buf);
    dev->d_lvl_simple_begin  = dev->d_sched_buf;
    dev->d_simple_state      = dev->d_lvl_simple_begin + n_levels + 1;
    dev->d_simple_item_start = dev->d_simple_state + n_simple;
    dev->d_lvl_il_begin      = dev->d_simple_item_start + n_simple + 1;
    dev->d_il_state          = dev->d_lvl_il_begin + n_levels + 1;
    dev->sched_n_levels = n_levels;
    if (getenv("GPU_PERF_CM_TIMING"))
        fprintf(stderr, "[LEVEL_SCHED]\tM=%d\tlevels=%d\tsimple=%d\til=%d\titems=%d\n",
                M, n_levels, n_simple, n_il, items);
    return 0;
}

void
gpu_cm_device_free(GPU_CM_Device *dev)
{
    if (!dev) return;
    gpu_dev_free(dev->d_begl_offsets); gpu_dev_free(dev->d_begl_lo); gpu_dev_free(dev->d_begl_width);
    gpu_dev_free(dev->d_sttype);   gpu_dev_free(dev->d_stid);
    gpu_dev_free(dev->d_cfirst);   gpu_dev_free(dev->d_cnum);
    gpu_dev_free(dev->d_emitmode); gpu_dev_free(dev->d_sd); gpu_dev_free(dev->d_sdr);
    gpu_dev_free(dev->d_tsc);      gpu_dev_free(dev->d_oesc); gpu_dev_free(dev->d_oesc_offset);
    gpu_dev_free(dev->d_beginsc);  gpu_dev_free(dev->d_endsc);
    gpu_dev_free(dev->d_init_sc);  gpu_dev_free(dev->d_begl_map);
    gpu_dev_free(dev->d_dmin1);    gpu_dev_free(dev->d_dmax1);
    gpu_dev_free(dev->d_sched_buf);
    dev->d_sched_buf = NULL; dev->sched_n_levels = 0;
    gpu_dev_free(dev->d_layout_buf);
    dev->d_layout_buf = NULL;
}

/* ──── Memory estimation and auto-tuning ──── */

size_t
gpu_workspace_bytes_per_window(int M, int W, int n_begl, int begl_packed_width)
{
    int W1 = W + 1;
    size_t alpha      = (size_t)2 * M * W1 * sizeof(float);
    size_t alpha_begl = (size_t)W1 * (begl_packed_width > 0 ? begl_packed_width : n_begl * W1) * sizeof(float);
    size_t bestsc     = (size_t)W1 * sizeof(float);
    size_t bestr      = (size_t)W1 * sizeof(int);
    size_t vsc        = sizeof(float);
    return alpha + alpha_begl + bestsc + bestr + vsc;
}

int
gpu_compute_max_batch(int M, int W, int n_begl, float mem_fraction, int begl_packed_width)
{
    size_t free_bytes = 0, total_bytes = 0;
    cudaError_t err = gpu_mem_get_info(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[gpu_autotune] cudaMemGetInfo failed: %s\n",
                cudaGetErrorString(err));
        return 128;
    }

    size_t per_win = gpu_workspace_bytes_per_window(M, W, n_begl, begl_packed_width);
    if (per_win == 0) return 1;

    if (mem_fraction <= 0.0f || mem_fraction > 1.0f) mem_fraction = 0.80f;
    size_t usable = (size_t)(free_bytes * mem_fraction);
    int max_batch = (int)(usable / per_win);
    if (max_batch < 1) max_batch = 1;

    static int autotune_count = 0;
    autotune_count++;
    if (autotune_count <= 3 || autotune_count % 500 == 0)
        fprintf(stderr, "[gpu_autotune] GPU mem: %.0f MB free / %.0f MB total, "
                "per_window: %.1f KB, max_batch: %d (%.0f%% of free)  (#%d)\n",
                free_bytes / 1e6, total_bytes / 1e6,
                per_win / 1024.0, max_batch, mem_fraction * 100, autotune_count);

    return max_batch;
}

size_t
gpu_workspace_bytes_full(const GPU_CM_Device *dev)
{
    size_t W1 = (size_t)dev->W + 1;
    return ((size_t)2 * dev->full_layout.a_width + (size_t)dev->full_layout.begl_floats) * sizeof(float)
         + W1 * sizeof(float) + W1 * sizeof(int) + sizeof(float);
}

int
gpu_compute_max_batch_full(const GPU_CM_Device *dev, float mem_fraction)
{
    size_t free_bytes = 0, total_bytes = 0;
    if (gpu_mem_get_info(&free_bytes, &total_bytes) != cudaSuccess) return 1;
    size_t per_win = gpu_workspace_bytes_full(dev);
    int n = (int)((size_t)(free_bytes * mem_fraction) / (per_win ? per_win : 1));
    return n < 1 ? 1 : n;
}

int
gpu_scan_workspace_create_full(GPU_ScanWorkspace *ws, int max_batch, const GPU_CM_Device *dev)
{
    memset(ws, 0, sizeof(*ws));
    ws->full_layout = 1;
    ws->max_batch = max_batch;
    ws->M = dev->M; ws->W = dev->W; ws->n_begl = dev->n_begl;
    size_t W1 = (size_t)dev->W + 1;
    CUDA_CHECK(gpu_dev_malloc(&ws->d_alpha, (size_t)max_batch * 2 * dev->full_layout.a_width * sizeof(float)));
    CUDA_CHECK(gpu_dev_malloc(&ws->d_alpha_begl, (size_t)max_batch * (size_t)(dev->full_layout.begl_floats > 0 ? dev->full_layout.begl_floats : 1) * sizeof(float)));
    CUDA_CHECK(gpu_dev_malloc(&ws->d_bestsc,   (size_t)max_batch * W1 * sizeof(float)));
    CUDA_CHECK(gpu_dev_malloc(&ws->d_bestr,    (size_t)max_batch * W1 * sizeof(int)));
    CUDA_CHECK(gpu_dev_malloc(&ws->d_vsc_root, (size_t)max_batch * sizeof(float)));
    return 0;
}

/* ──── Workspace allocation ──── */

int
gpu_scan_workspace_create(GPU_ScanWorkspace *ws, int max_batch, int M, int W, int n_begl, int begl_packed_width)
{
    int W1 = W + 1;
    ws->full_layout = 0;
    ws->begl_packed_width = begl_packed_width;
    ws->max_batch = max_batch;
    ws->M = M;
    ws->W = W;
    ws->n_begl = n_begl;

    size_t alpha_total      = (size_t)max_batch * 2 * M * W1 * sizeof(float);
    size_t alpha_begl_total = (size_t)max_batch * W1 * (begl_packed_width > 0 ? begl_packed_width : n_begl * W1) * sizeof(float);
    size_t bestsc_total     = (size_t)max_batch * W1 * sizeof(float);
    size_t bestr_total      = (size_t)max_batch * W1 * sizeof(int);
    size_t vsc_total        = (size_t)max_batch * sizeof(float);

    size_t total = alpha_total + alpha_begl_total + bestsc_total + bestr_total + vsc_total;
    static int ws_count = 0;
    ws_count++;
    if (ws_count <= 3 || ws_count % 500 == 0)
        fprintf(stderr, "[gpu_workspace] %.1f MB total (alpha: %.1f + begl: %.1f), batch=%d  (#%d)\n",
                total / 1e6, alpha_total / 1e6, alpha_begl_total / 1e6, max_batch, ws_count);

    CUDA_CHECK(gpu_dev_malloc(&ws->d_alpha,      alpha_total));
    CUDA_CHECK(gpu_dev_malloc(&ws->d_alpha_begl, alpha_begl_total));
    CUDA_CHECK(gpu_dev_malloc(&ws->d_bestsc,     bestsc_total));
    CUDA_CHECK(gpu_dev_malloc(&ws->d_bestr,      bestr_total));
    CUDA_CHECK(gpu_dev_malloc(&ws->d_vsc_root,   vsc_total));

    return 0;
}

void
gpu_scan_workspace_destroy(GPU_ScanWorkspace *ws)
{
    if (!ws) return;
    gpu_dev_free(ws->d_alpha);
    gpu_dev_free(ws->d_alpha_begl);
    gpu_dev_free(ws->d_bestsc);
    gpu_dev_free(ws->d_bestr);
    gpu_dev_free(ws->d_vsc_root);
}
