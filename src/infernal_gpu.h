#ifndef INFERNAL_GPU_H
#define INFERNAL_GPU_H

/*
 * infernal_gpu.h — Public API for GPU-accelerated Infernal CM scanning.
 *
 * Usage:
 *   1. Create GPU_CM_Host from your CM parameters (gpu_cm_host_create)
 *   2. Upload to GPU (gpu_cm_upload)
 *   3. Create workspace (gpu_scan_workspace_create)
 *   4. Call gpu_batch_cyk_scan() for each batch of windows
 *   5. Cleanup
 *
 * This header does not depend on Infernal headers; all types are self-contained.
 */

#include "gpu_types.h"

/* ── Host CM creation / destruction ── */

GPU_CM_Host *gpu_cm_host_create(
    int M, int W, int abc_Kp, float el_selfsc, int has_local_begins,
    const int8_t *sttype, const int8_t *stid,
    const int *cfirst, const int *cnum,
    const float *tsc_flat,
    const float *oesc_flat, const int *oesc_offset, int oesc_total,
    const float *beginsc, const float *endsc,
    const float *init_sc_flat,
    const int *dmin1, const int *dmax1   /* QDB bands; NULL → use sd[v]/W */
);

void gpu_cm_host_destroy(GPU_CM_Host *h);

/* ── Device upload / free ── */

int  gpu_cm_upload(const GPU_CM_Host *h, GPU_CM_Device *dev);
void gpu_cm_device_free(GPU_CM_Device *dev);

/* ── Memory estimation and auto-tuning ── */

/* Returns bytes of GPU workspace needed per window for a given CM. */
size_t gpu_workspace_bytes_per_window(int M, int W, int n_begl, int begl_packed_width = 0);

/* Queries GPU free memory and returns the maximum batch size that fits.
 * mem_fraction: fraction of free GPU memory to use (default 0.80). */
/* Fully packed layout (per-state alpha bands + short BEGL rings): per-window
 * bytes, batch capacity, and workspace creation. Kernels select the packed
 * instantiation from ws->full_layout. Requires a build with IL_PARALLEL_SCAN,
 * ROOT_STATE_FUSED and BIF_RING_INDEX (gpu_full_layout_supported()). */
size_t gpu_workspace_bytes_full(const GPU_CM_Device *dev);
int    gpu_compute_max_batch_full(const GPU_CM_Device *dev, float mem_fraction);
int    gpu_scan_workspace_create_full(GPU_ScanWorkspace *ws, int max_batch, const GPU_CM_Device *dev);
int    gpu_full_layout_supported(void);

int gpu_inside_resident_blocks(void);
int gpu_dp_resident_blocks(int cyk);   /* cyk=1: fast CYK kernel */

/* Fast CYK: max-plus instantiation of the optimized Inside kernel (IL prefix
 * max-scan, optional level schedule, packed BEGL). Bands come from the device
 * model, so upload a beta1 (tight) model to mirror Infernal's CYK filter.
 * alpha0 requires n_windows <= ws->max_batch. dmin_j (with bestsc_j), when
 * given, receives per end position the smallest root length d whose score is
 * >= dmin_cut (0x7fffffff if none or if the build lacks ROOT_STATE_FUSED). */
int gpu_batch_cyk_fast_scan_perj(const GPU_CM_Device *dev, GPU_ScanWorkspace *ws,
                                 const GPU_Window *windows, int n_windows,
                                 float *scores, float *bestsc_j, int perj_stride,
                                 float dmin_cut = 0.0f, int *dmin_j = NULL,
                                 const GPU_Null3 *n3 = NULL);
int gpu_batch_cyk_fast_scan_alpha0(const GPU_CM_Device *dev, GPU_ScanWorkspace *ws,
                                   const GPU_Window *windows, int n_windows,
                                   float *scores, float *alpha0_jd, int perj_stride, int alpha0_dstride);

int gpu_compute_max_batch(int M, int W, int n_begl, float mem_fraction, int begl_packed_width = 0);

/* ── Workspace ── */

/* Optional begl_packed_width must be 0 (dense) or dev.begl_packed_width.
 * Packed workspaces are Inside-only; gpu_batch_cyk_scan rejects them. */

int  gpu_scan_workspace_create(GPU_ScanWorkspace *ws, int max_batch, int M, int W, int n_begl, int begl_packed_width = 0);
void gpu_scan_workspace_destroy(GPU_ScanWorkspace *ws);

/* ── Batch CYK scan ── */

/* Scans n_windows windows. If n_windows > ws->max_batch, automatically
 * splits into sub-batches (no manual chunking needed by caller). */
int gpu_batch_cyk_scan(
    const GPU_CM_Device *dev,
    GPU_ScanWorkspace   *ws,
    const GPU_Window    *windows,
    int                  n_windows,
    float               *scores       /* [n_windows] output */
);

/* ── Batch Inside scan ── */

/* Inside algorithm (sum over all parses) instead of CYK (best parse).
 * Same interface as gpu_batch_cyk_scan. Inside scores are needed for
 * E-value computation, matching Infernal's default cmsearch behavior. */
int gpu_batch_inside_scan(
    const GPU_CM_Device *dev,
    GPU_ScanWorkspace   *ws,
    const GPU_Window    *windows,
    int                  n_windows,
    float               *scores       /* [n_windows] output */
);

/* ── Batch Inside scan WITH per-end-position bestsc[j] (for calibration) ──
 * In addition to the per-window max Inside score, returns bestsc[j] = best
 * local-Inside hit score ENDING at position j, for every j of every window.
 * This is exactly the per-j vector cmcalibrate feeds to its gamma semi-HMM
 * hit resolver. ALL windows must have the same length == perj_stride.
 * bestsc_j must be caller-allocated [n_windows * perj_stride]; unfilled cells
 * (j beyond a window's length) are left at GPU_IMPOSSIBLE. */
int gpu_batch_inside_scan_perj(
    const GPU_CM_Device *dev,
    GPU_ScanWorkspace   *ws,
    const GPU_Window    *windows,
    int                  n_windows,
    float               *scores,      /* [n_windows] output */
    float               *bestsc_j,    /* [n_windows * perj_stride] output */
    int                  perj_stride, /* >= every window length */
    const GPU_Null3     *n3 = NULL    /* optional NULL3-corrected per-j gate scores */
);

/* ── Inside scan emitting the FULL per-(j,d) root score matrix ──
 * alpha0_jd[((win*perj_stride)+(j-1))*alpha0_dstride + d] = best local-Inside
 * score of a hit of length d ending at position j (== cmcalibrate's per-length
 * bestsc[d] for ALL d). Lets the exact gamma semi-HMM hit resolver run on host.
 * All windows must be length == perj_stride; alpha0_dstride should be W+1.
 * Memory: n_windows*perj_stride*alpha0_dstride floats — keep n_windows modest. */
int gpu_batch_inside_scan_alpha0(
    const GPU_CM_Device *dev,
    GPU_ScanWorkspace   *ws,
    const GPU_Window    *windows,
    int                  n_windows,
    float               *scores,
    float               *alpha0_jd,
    int                  perj_stride,
    int                  alpha0_dstride
);

/* ── E-value computation ── */

/* Convert an Inside score to an E-value using the CM's calibration parameters.
 * db_residues = total database size in nucleotides (one strand).
 * n_models = number of CM models in the scan (for multiple testing correction).
 * Returns -1.0 if calibration parameters are not available. */
float gpu_evalue_from_score(
    float score,
    const GPU_EvalueParams *evp,
    int64_t db_residues,
    int n_models
);

#endif /* INFERNAL_GPU_H */
