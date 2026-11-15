/*
 * gpu_arena.h — optional whole-run device-memory reservation.
 *
 * On a shared server, a timing run is only clean if nobody else uses the
 * card meanwhile, and the scanner sizes its batches from the free device
 * memory it sees before each model. Both problems go away if the scanner
 * takes the card's memory once at start-up and serves every later device
 * allocation from that reservation:
 *   - other users cannot start a job on the card (no memory left for them);
 *   - batch sizes no longer depend on what other processes hold;
 *   - no cudaMalloc happens mid-run, so a neighbour's allocation between our
 *     cudaMemGetInfo and cudaMalloc can no longer make a model fail (OOM).
 *
 * Environment (read once by gpu_arena_init_from_env):
 *   GPU_MEM_RESERVE      unset/0: off (plain cudaMalloc, previous behaviour).
 *                        1 or auto: reserve all free memory minus the headroom.
 *                        N > 1: reserve N MB (a fixed budget, e.g. for tests).
 *   GPU_MEM_HEADROOM_MB  memory left to the driver in auto mode (kernel local
 *                        memory, module code, printf buffers); default 1024.
 *
 * In auto mode gpu_mem_get_info() reports (largest free block + headroom) as
 * free memory, which equals what an empty card reported before, so batch
 * sizes, tiling and therefore timings match the earlier exclusive-card runs.
 * Allocations that do not fit fall back to cudaMalloc (logged), so the arena
 * can only add memory, never lose an allocation that would have succeeded.
 *
 * Header-only: the arena lives in a function-local static of an inline
 * function, shared by all translation units that include this file.
 */
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <iterator>
#include <mutex>

struct GpuArena {
    char  *base = nullptr;
    size_t cap = 0, headroom = 0;
    bool   auto_mode = false;
    std::map<size_t, size_t> free_blocks;   /* offset -> size, coalesced */
    std::map<size_t, size_t> used;          /* offset -> size */
    std::map<void *, size_t> outside;       /* fallback cudaMalloc blocks */
    size_t in_use = 0, peak = 0;
    long   n_fallback = 0;
    std::mutex mu;
};

inline GpuArena &gpu_arena() { static GpuArena a; return a; }

inline bool gpu_arena_active() { return gpu_arena().base != nullptr; }

inline size_t gpu_arena_largest_free_locked(const GpuArena &a)
{
    size_t best = 0;
    for (const auto &b : a.free_blocks) if (b.second > best) best = b.second;
    return best;
}

inline void gpu_arena_report_atexit()
{
    GpuArena &a = gpu_arena();
    if (!a.base) return;
    fprintf(stderr, "[GPU_MEM_RESERVE] end: arena %.0f MB, peak use %.0f MB, fallback allocations %ld\n",
            a.cap / 1e6, a.peak / 1e6, a.n_fallback);
}

/* Reserve the arena if GPU_MEM_RESERVE asks for it. Returns 0 when off or
 * reserved, -1 if the reservation could not be made (the run continues with
 * plain cudaMalloc). Call once, after the device is selected. */
inline int gpu_arena_init_from_env()
{
    GpuArena &a = gpu_arena();
    if (a.base) return 0;
    const char *env = getenv("GPU_MEM_RESERVE");
    if (!env || !*env || strcmp(env, "0") == 0) return 0;
    size_t headroom = (size_t)1024 << 20;
    if (const char *h = getenv("GPU_MEM_HEADROOM_MB")) headroom = (size_t)atol(h) << 20;

    cudaFree(0);                                  /* create the context first */
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return -1;
    bool auto_mode = strcmp(env, "1") == 0 || strcmp(env, "auto") == 0;
    size_t want;
    if (auto_mode) {
        want = free_b > headroom ? free_b - headroom : 0;
    } else {
        want = (size_t)atol(env) << 20;
        if (want > free_b) want = free_b > headroom ? free_b - headroom : 0;
    }
    want &= ~(size_t)((2u << 20) - 1);            /* 2 MB granularity */
    void *p = nullptr;
    while (want >= ((size_t)256 << 20)) {
        if (cudaMalloc(&p, want) == cudaSuccess) break;
        cudaGetLastError();
        p = nullptr;
        want -= (size_t)256 << 20;                /* a neighbour moved; retry smaller */
    }
    if (!p) {
        fprintf(stderr, "[GPU_MEM_RESERVE] could not reserve device memory (free %.0f MB); "
                        "continuing without a reservation\n", free_b / 1e6);
        return -1;
    }
    a.base = (char *)p;
    a.cap = want;
    a.headroom = auto_mode ? headroom : 0;
    a.auto_mode = auto_mode;
    a.free_blocks[0] = want;
    fprintf(stderr, "[GPU_MEM_RESERVE] reserved %.0f MB of %.0f MB (free at start %.0f MB, %s headroom %.0f MB)\n",
            want / 1e6, total_b / 1e6, free_b / 1e6, auto_mode ? "auto," : "fixed budget,", headroom / 1e6);
    atexit(gpu_arena_report_atexit);
    return 0;
}

inline cudaError_t gpu_arena_malloc(void **p, size_t n)
{
    GpuArena &a = gpu_arena();
    if (!a.base) return cudaMalloc(p, n);
    size_t need = (n + 255) & ~(size_t)255;
    if (need == 0) need = 256;
    {
        std::lock_guard<std::mutex> lk(a.mu);
        auto best = a.free_blocks.end();
        for (auto it = a.free_blocks.begin(); it != a.free_blocks.end(); ++it)
            if (it->second >= need && (best == a.free_blocks.end() || it->second < best->second)) best = it;
        if (best != a.free_blocks.end()) {
            size_t off = best->first, sz = best->second;
            a.free_blocks.erase(best);
            if (sz > need) a.free_blocks[off + need] = sz - need;
            a.used[off] = need;
            a.in_use += need;
            if (a.in_use > a.peak) a.peak = a.in_use;
            *p = a.base + off;
            return cudaSuccess;
        }
    }
    cudaError_t e = cudaMalloc(p, n);
    std::lock_guard<std::mutex> lk(a.mu);
    if (e == cudaSuccess) {
        a.outside[*p] = n;
        if (a.n_fallback++ < 5)
            fprintf(stderr, "[GPU_MEM_RESERVE] %.1f MB did not fit in the arena (largest free %.1f MB); "
                            "allocated outside\n", n / 1e6, gpu_arena_largest_free_locked(a) / 1e6);
    }
    return e;
}

inline cudaError_t gpu_arena_free(void *p)
{
    GpuArena &a = gpu_arena();
    if (!p) return cudaSuccess;
    if (!a.base) return cudaFree(p);
    std::lock_guard<std::mutex> lk(a.mu);
    char *c = (char *)p;
    if (c < a.base || c >= a.base + a.cap) {
        a.outside.erase(p);
        return cudaFree(p);
    }
    size_t off = (size_t)(c - a.base);
    auto u = a.used.find(off);
    if (u == a.used.end()) return cudaErrorInvalidValue;
    size_t sz = u->second;
    a.used.erase(u);
    a.in_use -= sz;
    auto next = a.free_blocks.lower_bound(off);
    if (next != a.free_blocks.end() && off + sz == next->first) {
        sz += next->second;
        next = a.free_blocks.erase(next);
    }
    if (next != a.free_blocks.begin()) {
        auto prev = std::prev(next);
        if (prev->first + prev->second == off) {
            prev->second += sz;
            return cudaSuccess;
        }
    }
    a.free_blocks[off] = sz;
    return cudaSuccess;
}

/* Free memory as the batch-size heuristics should see it. */
inline cudaError_t gpu_mem_get_info(size_t *free_b, size_t *total_b)
{
    GpuArena &a = gpu_arena();
    if (!a.base) return cudaMemGetInfo(free_b, total_b);
    size_t f = 0, t = 0;
    cudaError_t e = cudaMemGetInfo(&f, &t);
    std::lock_guard<std::mutex> lk(a.mu);
    *free_b = gpu_arena_largest_free_locked(a) + a.headroom;
    *total_b = t;
    return e;
}

/* Largest free block in the arena (0 when no arena), for logging. */
inline size_t gpu_arena_free_bytes()
{
    GpuArena &a = gpu_arena();
    if (!a.base) return 0;
    std::lock_guard<std::mutex> lk(a.mu);
    return gpu_arena_largest_free_locked(a);
}

template <typename T>
inline cudaError_t gpu_dev_malloc(T **p, size_t n) { return gpu_arena_malloc((void **)p, n); }
inline cudaError_t gpu_dev_free(void *p) { return gpu_arena_free(p); }
