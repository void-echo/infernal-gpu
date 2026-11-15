/*
 * scan.cu — Infernal-GPU scanner (built as infernal-gpu-scan, and with
 * -DCALIB_RECORDS as infernal-gpu-calibrate-scan): scan sequences against
 * a library of CMs (e.g., Rfam.cm).
 *
 * For each CM in the library, uploads it to GPU, runs Inside scan on
 * all windows of the query sequence, computes E-values, and reports
 * hits.  Times the entire process.
 *
 * Usage:
 *   infernal-gpu-scan <cm_library> <fasta_file> [options]
 *
 * Options:
 *   --cyk               Use CYK instead of Inside
 *   --max-cm N          Process only the first N CMs (for quick testing)
 *   --hits-out FILE     Write all significant hits to FILE (TSV format)
 *   --evalue-cutoff F   E-value threshold for --hits-out (default: no filter)
 *   --candidate-evalue F  Gate per-j raw scores BEFORE region merging (Inside only)
 *   --evalue-models N   Original library size for subset/shard E-value scaling
 *   --cascade-cyk-p P   GPU CYK (beta1) filter at P, then Inside on survivors
 *   --null3-gate        Gate endpoints on NULL3-corrected scores
 *   --cyk-margin B      Bits by which the CYK and Inside gates are lowered (1.0)
 *   --hits-stream DIR   Also write each model's candidates to DIR as soon as
 *                       they exist (NNNNNN_<model>.tsv), then DIR/DONE
 *   --calibrate FILE    E-value calibration (cmcalibrate protocol) of every CM on
 *                       the given random sequences; writes one line per CM and
 *                       mode (ECMGC/ECMGI/ECMLC/ECMLI) to FILE
 *   --calib-budget N    Root cells exported per exact calibration pass (1.6e7)
 *   --gtailn N, --ltailn N  Hits/Mb fitted in glocal/local modes (cmcalibrate
 *                       defaults 250 and 750)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <functional>
#include <vector>
#include <map>
#include <thread>
#include <atomic>
#include <string>
#include <time.h>
#include <cuda_runtime.h>
#include "infernal_gpu.h"
#include "cm_file_parser.h"
#include "gpu_arena.h"

static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ── FASTA reader (same as validate_vs_infernal.cu) ── */

struct FastaSeq {
    std::string name;
    std::vector<uint8_t> dsq;
    int n;
};

static int char_to_digital_cs(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'U': case 'u': case 'T': case 't': return 3;
        case 'R': case 'r': return 5;
        case 'Y': case 'y': return 6;
        case 'M': case 'm': return 7;
        case 'K': case 'k': return 8;
        case 'S': case 's': return 9;
        case 'W': case 'w': return 10;
        case 'H': case 'h': return 11;
        case 'B': case 'b': return 12;
        case 'V': case 'v': return 13;
        case 'D': case 'd': return 14;
        case 'N': case 'n': case 'X': case 'x': return 15;
        default: return 15;
    }
}

static std::vector<FastaSeq> read_fasta_cs(const char *path) {
    std::vector<FastaSeq> seqs;
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "Cannot open FASTA: %s\n", path); return seqs; }

    char line[65536];
    std::string current_name, current_seq;

    auto flush = [&]() {
        if (current_seq.empty()) return;
        FastaSeq fs;
        fs.name = current_name;
        fs.n = (int)current_seq.size();
        fs.dsq.resize(fs.n + 2);
        fs.dsq[0] = 127;
        for (int i = 0; i < fs.n; i++)
            fs.dsq[i + 1] = (uint8_t)char_to_digital_cs(current_seq[i]);
        fs.dsq[fs.n + 1] = 127;
        seqs.push_back(fs);
    };

    while (fgets(line, sizeof(line), fp)) {
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';
        if (line[0] == '>') { flush(); current_name = line + 1; current_seq.clear(); }
        else current_seq += line;
    }
    flush();
    fclose(fp);
    return seqs;
}

/* ── Build overlapping windows from a sequence for a given CM ── */

struct ScanHit {
    std::string cm_name;
    std::string cm_acc;
    std::string seq_name;
    int seq_idx;
    int64_t i0, j0;
    float score;
    float evalue;
    char strand;  /* '+' or '-' */
    bool gate_export; /* passed a margin-lowered endpoint gate: exported
                       * regardless of --evalue-cutoff, which would undo it */
};

/* Candidate TSV writer shared by --hits-out and --hits-stream. */
static int write_hits_tsv(FILE *hfp, const ScanHit *begin, const ScanHit *end, float evalue_cutoff) {
    int n_written = 0;
    fprintf(hfp, "#target_name\tcm_name\tcm_acc\tseq_from\tseq_to\tstrand\tscore\tevalue\n");
    for (const ScanHit *h = begin; h != end; ++h) {
        if (evalue_cutoff < 0 || h->evalue <= evalue_cutoff || h->gate_export) {
            fprintf(hfp, "%s\t%s\t%s\t%ld\t%ld\t%c\t%.4f\t%.6e\n",
                    h->seq_name.c_str(), h->cm_name.c_str(), h->cm_acc.c_str(),
                    (long)h->i0, (long)h->j0, h->strand, h->score, h->evalue);
            n_written++;
        }
    }
    return n_written;
}

/* ── Reverse complement of a digital sequence ── */
/* Complement table for Infernal's digital encoding:
   0=A 1=C 2=G 3=U → complement: A↔U(3↔0), C↔G(1↔2)
   For ambiguity codes (4-15): complement symmetrically. */
static uint8_t complement_digital(uint8_t c) {
    /* AUDIT 2026-06-25: the previous table was written
     * for a wrong alphabet ordering and silently corrupted minus-strand windows
     * containing true IUPAC degenerate codes (R/Y/M/K/W/H/D — 7/11 entries wrong,
     * incl. R→gap). Corrected to match the Easel RNA alphabet "ACGU-RYMKSWHBVDN"
     * used by char_to_digital_cs() above. Complement pairs:
     *   0:A↔3:U   1:C↔2:G   4:gap (self)
     *   5:R(AG)↔6:Y(CU)   7:M(AC)↔8:K(GU)   9:S(GC) self   10:W(AU) self
     *   11:H(ACU)↔14:D(AGU)   12:B(CGU)↔13:V(ACG)   15:N self */
    static const uint8_t comp[16] = {
        3, 2, 1, 0,        /* A C G U */
        4,                 /* gap */
        6, 5, 8, 7,        /* R Y M K */
        9, 10,             /* S W */
        14, 13, 12, 11,    /* H B V D */
        15                 /* N */
    };
    return (c < 16) ? comp[c] : 15;
}

static FastaSeq reverse_complement(const FastaSeq &fs) {
    FastaSeq rc;
    rc.name = fs.name;
    rc.n = fs.n;
    rc.dsq.resize(fs.n + 2);
    rc.dsq[0] = 127;                          /* sentinel */
    for (int i = 0; i < fs.n; i++)
        rc.dsq[i + 1] = complement_digital(fs.dsq[fs.n - i]);  /* reverse + complement */
    rc.dsq[fs.n + 1] = 127;                   /* sentinel */
    return rc;
}

/* ── Tiling over sequence segments ──
 * A segment is an inclusive [lo,hi] range of one sequence strand, in the
 * coordinates of that strand's digital sequence (reverse complement for '-').
 * W/2 Plan B tiles a segment with length (C+1)*W and stride C*W; every tile
 * after the first discards its first W positions (warm-up), so each end
 * position of the segment is reported exactly once and sees all lengths <= W
 * that stay inside the segment. */
struct Segment { int seq_idx; char strand; int64_t lo, hi; };
struct WinMeta { int seq_idx; int seq_len; char strand;
                 int64_t s;        /* tile start in cur_seq (1-based) */
                 int win_len;      /* tile length (residues)          */
                 int trust_lo_jp;  /* first trustworthy window-local jp */ };

static int64_t count_tiles(const std::vector<Segment> &segs, int64_t C, int W) {
    int64_t n_tiles = 0, t = (C + 1) * W, st = C * W;
    for (const Segment &g : segs) {
        int64_t n = g.hi - g.lo + 1;
        n_tiles += (n <= t) ? 1 : 1 + (n - t + st - 1) / st;
    }
    return n_tiles;
}

/* Wave-aware overlap factor: one block scans one tile sequentially, so a
 * launch lasts about (#waves) x (tile length). Minimize waves*(C+1) subject to
 * tiles <= waves * cap, where cap = concurrently runnable blocks. */
static int choose_wave_c(const std::vector<Segment> &segs, int W, int64_t cap, int64_t *waves_out) {
    int64_t n_max = 1;
    for (const Segment &g : segs) n_max = std::max(n_max, g.hi - g.lo + 1);
    int64_t c_max = std::max((int64_t)1, (n_max + W - 1) / W);
    int64_t best_c = 8, best_cost = -1, best_w = 0;
    for (int64_t w = 1; w <= 256; w++) {
        if (count_tiles(segs, c_max, W) > w * cap) continue;
        int64_t lo = 1, hi = c_max;
        while (lo < hi) { int64_t mid = (lo + hi) / 2; if (count_tiles(segs, mid, W) <= w * cap) hi = mid; else lo = mid + 1; }
        if (best_cost < 0 || w * (lo + 1) < best_cost) { best_cost = w * (lo + 1); best_c = lo; best_w = w; }
    }
    if (waves_out) *waves_out = best_w;
    return (int)best_c;
}

static void build_tiles(const std::vector<Segment> &segs, int C, int W,
                        const std::vector<FastaSeq> &fwd, const std::vector<FastaSeq> &rev,
                        std::vector<GPU_Window> &wins, std::vector<WinMeta> &meta) {
    int64_t stride = std::max((int64_t)1, (int64_t)C * W);
    int64_t tile   = std::max((int64_t)1, (int64_t)(C + 1) * W);
    wins.clear(); meta.clear();
    for (const Segment &g : segs) {
        const FastaSeq &cur = (g.strand == '+') ? fwd[g.seq_idx] : rev[g.seq_idx];
        for (int64_t s = g.lo; s <= g.hi; s += stride) {
            GPU_Window win;
            win.dsq = cur.dsq.data();
            win.i0 = s;
            win.j0 = std::min(s + tile - 1, g.hi);
            win.cutoff = GPU_IMPOSSIBLE;
            wins.push_back(win);
            WinMeta m;
            m.seq_idx = g.seq_idx; m.seq_len = cur.n; m.strand = g.strand;
            m.s = s; m.win_len = (int)(win.j0 - win.i0 + 1);
            m.trust_lo_jp = (s == g.lo) ? 1 : (W + 1);
            meta.push_back(m);
            if (win.j0 >= g.hi) break;
        }
    }
}

/* ── E-value calibration (--calibrate) ──────────────────────────────────────
 * Reproduces Infernal's cmcalibrate hit collection on the host from root
 * cells exported by the GPU: ReportHitsGreedily per end position (lengths
 * ascending, NULL3-corrected, strict running maximum of reported scores),
 * then greedy overlap removal per sequence (score high to low, start low to
 * high). */
struct CalCand { float score; int start, stop; };
struct CalCell { int seq, j, d; float sc; };

static float calib_null3(const double *hi, const double *lo, int d, const float *null0, float omega) {
    float comp[4], tot = 0.0f;
    for (int a = 0; a < 4; a++) { comp[a] = (float)(hi[a] - lo[a]); if (comp[a] < 0.0f) comp[a] = 0.0f; tot += comp[a]; }
    float score = 0.0f;
    for (int a = 0; a < 4; a++) {
        float c = (tot > 0.0f) ? comp[a] / tot : 0.25f;                  /* esl_vec_FNorm */
        if (c > 0.0f) score += log2f(c / null0[a]) * c * (float)d;
    }
    score += log2f(omega);
    float mx = std::max(0.0f, score), mn = std::min(0.0f, score);        /* LogSum2(0, score) */
    return (mx - mn > 50.0f) ? mx : mx + log2f(1.0f + powf(2.0f, mn - mx));
}

static void calib_resolve(std::vector<CalCand> &cand, int L, std::vector<char> &covered, std::vector<float> &scores) {
    (void)L; (void)covered;
    std::sort(cand.begin(), cand.end(), [](const CalCand &a, const CalCand &b) {
        if (a.score != b.score) return a.score > b.score;
        return a.start < b.start;
    });
    /* Accepted hits never overlap, so only the accepted hit with the largest
     * start <= c.stop can overlap c. */
    std::map<int, int> acc;
    for (const CalCand &c : cand) {
        auto it = acc.upper_bound(c.stop);
        if (it != acc.begin() && std::prev(it)->second >= c.start) continue;
        acc.emplace(c.start, c.stop);
        scores.push_back(c.score);
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <cm_library> <fasta_file> [--cyk] [--max-cm N]\n", argv[0]);
        return 1;
    }

    const char *cm_lib_path  = argv[1];
    const char *fasta_path   = argv[2];
    int use_cyk = 0;
    int max_cm  = 0;  /* 0 = unlimited */
    const char *hits_out_path = NULL;
    const char *hits_stream_dir = NULL; /* --hits-stream DIR: per-model candidate files */
    float evalue_cutoff = -1.0f; /* <0 means no filter */
    double candidate_evalue = -1.0;
    int evalue_models = 0;
    float min_score = 0.0f; /* candidate-window score gate; lower to retain
                             * borderline windows for pipeline-equivalence study */
    double cascade_p = -1.0;     /* --cascade-cyk-p: CYK filter P threshold (Infernal F6) */
    double cyk_margin = 1.0;     /* --cyk-margin: bits subtracted from the CYK and Inside gates */
    int null3_gate = 0;          /* --null3-gate: gate on NULL3-corrected scores (margin applies) */
    int64_t evalue_residues = 0; /* --evalue-residues: single-strand residues of the original search
                                  * space when scanning extracted segments (E = 2*L*models scale) */
    int top_only = 0;            /* --toponly: scan given (pre-oriented) sequences on + strand only */
    int model_prefix = 0;        /* --model-prefix: each CM scans only sequences named "<CM name>|..."
                                  * (many per-model segment sets in one process) */
    const char *calib_out_path = NULL;  /* --calibrate FILE */
    long calib_budget = 16000000L;      /* --calib-budget: root cells per exact pass */
    double calib_gtailn = 250.0, calib_ltailn = 750.0;  /* cmcalibrate --gtailn / --ltailn */
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--cyk") == 0) use_cyk = 1;
        if (strcmp(argv[i], "--max-cm") == 0 && i + 1 < argc) max_cm = atoi(argv[++i]);
        if (strcmp(argv[i], "--hits-out") == 0 && i + 1 < argc) hits_out_path = argv[++i];
        if (strcmp(argv[i], "--hits-stream") == 0 && i + 1 < argc) hits_stream_dir = argv[++i];
        if (strcmp(argv[i], "--evalue-cutoff") == 0 && i + 1 < argc) evalue_cutoff = (float)atof(argv[++i]);
        if (strcmp(argv[i], "--min-score") == 0 && i + 1 < argc) min_score = (float)atof(argv[++i]);
        if (strcmp(argv[i], "--candidate-evalue") == 0 && i + 1 < argc) candidate_evalue = atof(argv[++i]);
        if (strcmp(argv[i], "--evalue-models") == 0 && i + 1 < argc) evalue_models = atoi(argv[++i]);
        if (strcmp(argv[i], "--cascade-cyk-p") == 0 && i + 1 < argc) cascade_p = atof(argv[++i]);
        if (strcmp(argv[i], "--cyk-margin") == 0 && i + 1 < argc) cyk_margin = atof(argv[++i]);
        if (strcmp(argv[i], "--null3-gate") == 0) null3_gate = 1;
        if (strcmp(argv[i], "--evalue-residues") == 0 && i + 1 < argc) evalue_residues = atoll(argv[++i]);
        if (strcmp(argv[i], "--toponly") == 0) top_only = 1;
        if (strcmp(argv[i], "--model-prefix") == 0) model_prefix = 1;
        if (strcmp(argv[i], "--calibrate") == 0 && i + 1 < argc) calib_out_path = argv[++i];
        if (strcmp(argv[i], "--calib-budget") == 0 && i + 1 < argc) calib_budget = atol(argv[++i]);
        if (strcmp(argv[i], "--gtailn") == 0 && i + 1 < argc) calib_gtailn = atof(argv[++i]);
        if (strcmp(argv[i], "--ltailn") == 0 && i + 1 < argc) calib_ltailn = atof(argv[++i]);
    }
    if (calib_out_path) {
        if (use_cyk || cascade_p != -1.0 || candidate_evalue != -1.0 || null3_gate || model_prefix || calib_budget < 1000000L
            || calib_gtailn < 100.0 || calib_ltailn < 100.0) {
            fprintf(stderr, "--calibrate runs alone (it scans all four modes itself); --calib-budget >= 1e6; --gtailn/--ltailn >= 100.\n");
            return 1;
        }
        top_only = 1;   /* calibration sequences are searched on the given strand only */
    }
    if (cascade_p != -1.0 && (use_cyk || !(cascade_p > 0.0 && cascade_p <= 1.0) || !(cyk_margin >= 0.0))) {
        fprintf(stderr, "--cascade-cyk-p needs Inside mode, P in (0,1] and a non-negative --cyk-margin.\n");
        return 1;
    }
    if (null3_gate && (use_cyk || !(candidate_evalue > 0.0) || !(cyk_margin >= 0.0))) {
        fprintf(stderr, "--null3-gate needs Inside mode and --candidate-evalue.\n");
        return 1;
    }
    if ((candidate_evalue != -1.0 && (!std::isfinite(candidate_evalue) || candidate_evalue <= 0.0))
        || evalue_models < 0 || (use_cyk && candidate_evalue > 0.0)) {
        fprintf(stderr, "Invalid candidate threshold/library size; --candidate-evalue requires Inside and E>0.\n");
        return 1;
    }

    if (use_cyk && getenv("GPU_PACKED_BEGL") &&
        (strcmp(getenv("GPU_PACKED_BEGL"), "1") == 0 || strcmp(getenv("GPU_PACKED_BEGL"), "auto") == 0)) {
        fprintf(stderr, "GPU_PACKED_BEGL is Inside-only and cannot be combined with --cyk.\n");
        return 1;
    }

    /* Print GPU info */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("=== Infernal-GPU cmscan Benchmark ===\n");
    printf("GPU: %s  (SM %d.%d, %.0f MB)\n",
           prop.name, prop.major, prop.minor, prop.totalGlobalMem / 1e6);
    printf("Algorithm: %s\n", use_cyk ? "CYK" : "Inside");
    printf("CM library: %s\n", cm_lib_path);
    printf("Query FASTA: %s\n\n", fasta_path);

    /* Optional whole-run device-memory reservation (GPU_MEM_RESERVE, see
     * src/gpu_arena.h): keeps the card to ourselves during timing runs and
     * makes batch sizes independent of other processes. */
    gpu_arena_init_from_env();

    /* Read query sequences */
    auto seqs = read_fasta_cs(fasta_path);
    if (seqs.empty()) {
        fprintf(stderr, "No sequences read from %s\n", fasta_path);
        return 1;
    }

    int64_t total_residues = 0;
    for (auto &s : seqs) total_residues += s.n;
    printf("Query: %d sequences, %ld total residues\n\n", (int)seqs.size(), (long)total_residues);
    const int64_t scanned_residues = total_residues;
    if (evalue_residues < 0) { fprintf(stderr, "--evalue-residues must be positive.\n"); return 1; }
    if (evalue_residues > 0) {
        total_residues = evalue_residues;  /* used only for E-value search space */
        printf("E-value search space: %ld residues per strand (scanned %ld)%s\n\n",
               (long)total_residues, (long)scanned_residues, top_only ? ", top strand only" : "");
    }

    /* Count CMs in library (for E-value multiple testing correction).
     * Each Infernal CM file has TWO `NAME` lines: one for the CM proper
     * (immediately after the `INFERNAL` magic line) and one for the
     * embedded HMM filter (later in the file). Counting all NAME lines
     * double-counts. Use a flag: only count NAME lines that
     * immediately follow an INFERNAL header. */
    int total_cm_count = 0;
    {
        FILE *count_fp = fopen(cm_lib_path, "r");
        if (count_fp) {
            char cbuf[256];
            int infernal_flag = 0;
            while (fgets(cbuf, sizeof(cbuf), count_fp)) {
                if (strncmp(cbuf, "INFERNAL", 8) == 0) {
                    infernal_flag = 1;
                } else if (infernal_flag && strncmp(cbuf, "NAME", 4) == 0) {
                    total_cm_count++;
                    infernal_flag = 0;
                }
            }
            fclose(count_fp);
        }
        if (max_cm > 0 && max_cm < total_cm_count) total_cm_count = max_cm;
        const int scanned_model_count = total_cm_count;
        if (evalue_models > 0) total_cm_count = evalue_models;
        printf("CM library: %d models (using %d for E-value correction)\n",
               scanned_model_count, total_cm_count);
        printf("Candidate E-value gate: %.9g (raw, pre-null3; -1 = legacy score gate)\n", candidate_evalue);
    }

    /* Open CM library file */
    FILE *cm_fp = fopen(cm_lib_path, "r");
    if (!cm_fp) {
        fprintf(stderr, "Cannot open CM library: %s\n", cm_lib_path);
        return 1;
    }

    /* ── Main loop: process CMs one at a time ── */
    int n_cms = 0;
    int n_hits = 0;
    double total_gpu_ms = 0.0;
    double total_upload_ms = 0.0;
    double total_wall_start = now_sec();

    std::vector<ScanHit> all_hits;

    std::vector<FastaSeq> rc_seqs(seqs.size());
    for (int si = 0; si < (int)seqs.size(); si++)
        rc_seqs[si] = reverse_complement(seqs[si]);
    std::vector<Segment> full_segs;
    for (int si = 0; si < (int)seqs.size(); si++)
        for (char strand : {'+', '-'})
            if (strand == '+' || !top_only)
                full_segs.push_back({si, strand, 1, (int64_t)seqs[si].n});

    /* Calibration: composition prefix counts per sequence (NULL3 on the host). */
    FILE *calib_fp = NULL;
    std::vector<std::vector<double> > calib_pre;
    if (calib_out_path) {
        calib_fp = fopen(calib_out_path, "w");
        if (!calib_fp) { fprintf(stderr, "Cannot write %s\n", calib_out_path); return 1; }
        fprintf(calib_fp, "#cm_name\tcm_acc\tmode\tlambda\tmu_extrap\tmu_orig\tdbsize\tnrandhits\ttailp\tn_fit\tnrand_exact\texact_seqs\tcut\tcells_exact\tcells_tail\tpasses\tseconds\n");
        calib_pre.resize(seqs.size());
        for (size_t si = 0; si < seqs.size(); si++) {
            std::vector<double> &pre = calib_pre[si];
            pre.assign(((size_t)seqs[si].n + 1) * 4, 0.0);
            for (int q = 1; q <= seqs[si].n; q++) {
                int x = seqs[si].dsq[q];
                for (int a = 0; a < 4; a++)
                    pre[(size_t)q * 4 + a] = pre[(size_t)(q - 1) * 4 + a] + (x < 4 ? (x == a ? 1.0 : 0.0) : 0.25);
            }
        }
    }

    while (1) {
        if (max_cm > 0 && n_cms >= max_cm) break;
        size_t hits_mark = all_hits.size();

        long cm_pos = ftell(cm_fp);
        /* Calibration uses the loose beta2 bands for every mode, as cmcalibrate does. */
        GPU_CM_Host *cm = parse_cm_from_fp_bands(cm_fp, 1, (cascade_p > 0.0 || calib_out_path) ? 2 : 0);  /* local mode */
        if (!cm) break;
        /* Cascade: the CYK filter uses Infernal's beta1 (tight) bands from the
         * same model record; the final Inside stage keeps beta2. */
        GPU_CM_Host *cm_cyk = NULL;
        if (cascade_p > 0.0) {
            long cm_end = ftell(cm_fp);
            fseek(cm_fp, cm_pos, SEEK_SET);
            cm_cyk = parse_cm_from_fp_bands(cm_fp, 1, 1);
            fseek(cm_fp, cm_end, SEEK_SET);
            if (!cm_cyk || cm_cyk->M != cm->M || cm_cyk->W != cm->W) {
                fprintf(stderr, "CM %s: failed to re-read beta1 bands\n", cm->name);
                return 1;
            }
        }
        n_cms++;
        std::vector<Segment> model_segs;
        const std::vector<Segment> *active_segs = &full_segs;
        if (model_prefix) {
            const std::string pre = std::string(cm->name) + "|";
            for (const Segment &sg : full_segs)
                if (seqs[sg.seq_idx].name.compare(0, pre.size(), pre) == 0) model_segs.push_back(sg);
            active_segs = &model_segs;
            if (model_segs.empty()) {
                gpu_cm_host_destroy(cm);
                if (cm_cyk) gpu_cm_host_destroy(cm_cyk);
                continue;
            }
        }

        /* ── Inter-CM quiescence barrier (W2 Plan B hang workaround) ──
         * On extended (≳50-CM) runs the Plan B perj scan path intermittently
         * busy-hangs the GPU on (deterministically) the 5th CM of the run. The
         * hang was isolated 2026-06-26 to a timing-
         * sensitive race in the perj scan path — NOT a memory OOB/leak (free mem
         * constant across CMs, compute-sanitizer memcheck clean, window bounds
         * verified ≤ perj_stride). It is reliably SUPPRESSED by a host-side
         * quiescence + small I/O round-trip between consecutive CMs (a single
         * isolated CUDA barrier OR error-clear OR cudaMemGetInfo was NOT enough;
         * the combination + a stderr write reproducibly fixes it across GPUs,
         * CM files, and 50/100-CM runs). Until the root cause is fixed in the
         * kernel, apply that barrier here. Set GPU_W2_NOFIX=1 to reproduce the
         * raw hang for debugging. */
        if (!getenv("GPU_W2_NOFIX")) {
            cudaDeviceSynchronize();
            cudaError_t pe = cudaGetLastError();
            size_t fb = 0, tb = 0; cudaMemGetInfo(&fb, &tb);
            if (gpu_arena_active())
                fprintf(stderr, "[W2_BARRIER] iter %d (%s W=%d): free=%.0f MB err=%s arena_free=%.0f MB\n",
                        n_cms, cm->name, cm->W, fb/1e6, cudaGetErrorString(pe), gpu_arena_free_bytes()/1e6);
            else
                fprintf(stderr, "[W2_BARRIER] iter %d (%s W=%d): free=%.0f MB err=%s\n",
                        n_cms, cm->name, cm->W, fb/1e6, cudaGetErrorString(pe));
        }

        /* Upload CM */
        double t_up0 = now_sec();
        GPU_CM_Device dev;
        memset(&dev, 0, sizeof(dev));
        GPU_CM_Device dev_cyk;
        memset(&dev_cyk, 0, sizeof(dev_cyk));
        if (gpu_cm_upload(cm, &dev) != 0 || (cm_cyk && gpu_cm_upload(cm_cyk, &dev_cyk) != 0)) {
            /* A skipped model would silently drop its hits: fail the run. */
            fprintf(stderr, "  [FATAL] CM %d (%s): upload failed\n", n_cms, cm->name);
            return 1;
        }
        double t_up1 = now_sec();
        total_upload_ms += (t_up1 - t_up0) * 1000.0;

        /* ── W/2 Plan B: large overlapping tiles + per-end-position output ──
         * Replaces the old stride-W/2 / one-max-per-window scheme, which (a)
         * deterministically missed hits of length d∈(W/2, W] and (b) wasted ~2×
         * compute. Tiles have length (C+1)*W and stride C*W with a W warm-up
         * (build_tiles), keeping the alpha workspace O(M·W) regardless of tile
         * length. We read bestsc[j] (best local hit score ENDING at j) for every
         * trustworthy end position, threshold, then merge nearby end positions
         * into candidate regions.
         *
         * Stage kinds: 0 = Inside (per-j), 1 = fast CYK (per-j), 2 = legacy
         * CYK (one max per tile). */
        int W = cm->W;
        int ws_batch = 0;
        auto run_stage = [&](const GPU_CM_Device &d, int kind, const std::vector<Segment> &segs,
                             std::vector<GPU_Window> &wins, std::vector<WinMeta> &meta,
                             std::vector<float> &scores, std::vector<float> &bj, int &stride,
                             float dmin_cut = 0.0f, std::vector<int> *dmin = NULL,
                             const GPU_Null3 *n3 = NULL) -> int {
            int C = 8;
            const char *c_env = getenv("GPU_W2_OVERLAP_C");
            const char *pk = getenv("GPU_PACKED_BEGL");
            bool may_pack = kind != 2 && pk && (strcmp(pk, "1") == 0 || strcmp(pk, "auto") == 0) && d.begl_packed_width > 0;
            int batch_cap = 0;
            if (const char *cap_env = getenv("GPU_W2_MAX_BATCH")) batch_cap = atoi(cap_env);
            /* Workspace layouts: 0 dense, 1 packed BEGL (GPU_PACKED_BEGL), 2 fully
             * packed alpha bands + short BEGL rings (GPU_PACK_ALL). "1" forces a
             * layout; "auto" takes the one that runs the most tiles at once
             * (min of resident blocks, memory fit and tile count), preferring the
             * simpler layout on ties since packed indexing costs a little. */
            const char *pa = getenv("GPU_PACK_ALL");
            bool may_full = kind != 2 && gpu_full_layout_supported() && pa && (strcmp(pa, "1") == 0 || strcmp(pa, "auto") == 0);
            int dense_fit = std::max(1, gpu_compute_max_batch(cm->M, cm->W, cm->n_begl, 0.75f));
            int packed_fit = may_pack ? std::max(1, gpu_compute_max_batch(cm->M, cm->W, cm->n_begl, 0.75f, d.begl_packed_width)) : 0;
            int full_fit = may_full ? std::max(1, gpu_compute_max_batch_full(&d, 0.75f)) : 0;
            if (batch_cap > 0) {
                dense_fit = std::min(dense_fit, batch_cap);
                if (packed_fit) packed_fit = std::min(packed_fit, batch_cap);
                if (full_fit) full_fit = std::min(full_fit, batch_cap);
            }
            int resident = kind != 2 ? gpu_dp_resident_blocks(kind == 1) : 0;
            auto concurrency = [&](int fit, int64_t n_tiles) {
                int64_t c = fit;
                if (resident > 0) c = std::min(c, (int64_t)resident);
                if (n_tiles > 0) c = std::min(c, n_tiles);
                return c;
            };
            if (c_env) { int v = atoi(c_env); if (v >= 1) C = v; }
            else if (kind != 2 && getenv("GPU_TILE_POLICY") && strcmp(getenv("GPU_TILE_POLICY"), "wave") == 0) {
                /* Wave-aware tiling; per-j scores do not depend on C. */
                int64_t cap = std::max(concurrency(dense_fit, 0), std::max(concurrency(packed_fit, 0), concurrency(full_fit, 0)));
                int64_t waves = 0;
                C = choose_wave_c(segs, W, cap, &waves);
                /* Bound the tile length: per-j outputs cost 8 bytes per tile
                 * position on the device for a whole batch, and float prefix
                 * counts (NULL3 gate) stay exact below 2^24. On large genomes
                 * this only adds waves, where quantization no longer matters. */
                C = std::min(C, std::max(8, (1 << 17) / W));
                if (getenv("GPU_PERF_CM_TIMING"))
                    fprintf(stderr, "[TILE_POLICY]\t%s\tkind=%d\tresident=%d\tmem_fit=%d\tC=%d\twaves=%ld\ttiles=%ld\n",
                            cm->name, kind, resident, std::max(dense_fit, std::max(packed_fit, full_fit)), C, (long)waves,
                            (long)count_tiles(segs, C, W));
            }
            build_tiles(segs, C, W, seqs, rc_seqs, wins, meta);
            int n = (int)wins.size();
            stride = (int)std::min((int64_t)(C + 1) * W, (int64_t)INT32_MAX);
            scores.assign(n, GPU_IMPOSSIBLE);
            bj.clear();
            if (n == 0) return 0;
            int layout = 0, max_fit = dense_fit;
            if (may_pack && (strcmp(pk, "1") == 0 || concurrency(packed_fit, n) > concurrency(max_fit, n))) {
                layout = 1; max_fit = packed_fit;
            }
            if (may_full && (strcmp(pa, "1") == 0 || concurrency(full_fit, n) > concurrency(max_fit, n))) {
                layout = 2; max_fit = full_fit;
            }
            int packed_width = layout == 1 ? d.begl_packed_width : 0;
            ws_batch = std::max(1, std::min(max_fit, n));
            GPU_ScanWorkspace ws = {};
            int ws_rc = layout == 2 ? gpu_scan_workspace_create_full(&ws, ws_batch, &d)
                                    : gpu_scan_workspace_create(&ws, ws_batch, cm->M, cm->W, cm->n_begl, packed_width);
            if (ws_rc != 0) {
                fprintf(stderr, "  [ERR] CM %s: workspace alloc failed (M=%d W=%d)\n", cm->name, cm->M, cm->W);
                return -1;
            }
            if (getenv("GPU_PERF_CM_TIMING"))
                fprintf(stderr, "[BEGL_LAYOUT]\t%s\tkind=%d\tlayout=%d\twindows=%d\tbatch=%d\tbytes_per_window=%zu\n",
                        cm->name, kind, layout, n, ws_batch,
                        layout == 2 ? gpu_workspace_bytes_full(&d)
                                    : gpu_workspace_bytes_per_window(cm->M, cm->W, cm->n_begl, packed_width));
            int stage_rc;
            if (kind == 2) {
                stage_rc = gpu_batch_cyk_scan(&d, &ws, wins.data(), n, scores.data());
            } else {
                /* Cells beyond a shorter (segment-end) tile's length are not
                 * meaningful; readers stop at win_len. */
                bj.assign((size_t)n * stride, GPU_IMPOSSIBLE);
                if (dmin) dmin->assign((size_t)n * stride, 0x7fffffff);
                stage_rc = (kind == 1)
                    ? gpu_batch_cyk_fast_scan_perj(&d, &ws, wins.data(), n, scores.data(), bj.data(), stride,
                                                   dmin_cut, dmin ? dmin->data() : NULL, n3)
                    : gpu_batch_inside_scan_perj(&d, &ws, wins.data(), n, scores.data(), bj.data(), stride, n3);
            }
            cudaDeviceSynchronize();
            gpu_scan_workspace_destroy(&ws);
            return stage_rc;
        };

        if (calib_fp) {
            /* Four modes in cmcalibrate's order: glocal CYK, glocal Inside,
             * local CYK, local Inside. Glocal modes re-read the model without
             * local begins/ends; all use the beta2 bands. */
            static const char *mode_name[4] = {"ECMGC", "ECMGI", "ECMLC", "ECMLI"};
            const long cm_end_pos = ftell(cm_fp);
            const int nseq = (int)seqs.size();
            const double total_res = (double)scanned_residues;
            for (int mode = 0; mode < 4; mode++) {
                const bool is_local = mode >= 2, is_inside = (mode & 1) != 0;
                const double tm0 = now_sec();
                GPU_CM_Host *cmm = cm;
                GPU_CM_Device devm = dev;
                if (!is_local) {
                    fseek(cm_fp, cm_pos, SEEK_SET);
                    cmm = parse_cm_from_fp_bands(cm_fp, 0, 2);
                    fseek(cm_fp, cm_end_pos, SEEK_SET);
                    memset(&devm, 0, sizeof(devm));
                    if (!cmm || cmm->M != cm->M || cmm->W != cm->W || gpu_cm_upload(cmm, &devm) != 0) {
                        fprintf(stderr, "  [FATAL] CM %s: glocal model setup failed\n", cm->name);
                        return 1;
                    }
                }
                const int kind = is_inside ? 0 : 1;
                const int rootlo = std::max(1, cmm->dmin1[0]);
                const int roothi = std::min(cmm->dmax1[0], W);
                const int band = std::max(1, roothi - rootlo + 1);
                const bool use_n3 = cmm->has_null3 != 0;
                const double tailn = is_local ? calib_ltailn : calib_gtailn;   /* cmcalibrate --ltailn / --gtailn */
                const float nhits_to_fit = (float)(tailn * (total_res / 1000000.0));

                /* Exact pass: export every root cell for the first P sequences. */
                int P = (int)std::min<int64_t>(nseq, std::max<int64_t>(1, calib_budget / std::max<int64_t>(1, (int64_t)(total_res / nseq) * band)));
                std::vector<float> exact_scores, tail_scores;
                static std::vector<int> cells;     /* reused across modes and models; never cleared */
                double tg_exact = 0.0, th_exact = 0.0, tg_tail = 0.0, th_tail = 0.0;
                std::vector<CalCand> cand;
                std::vector<char> covered;
                long cells_exact = 0, cells_tail = 0;
                int passes = 0;
                double exact_res = 0.0;
#ifdef CALIB_RECORDS
                /* One pass over every sequence. The GPU emits only root cells that
                 * ReportHitsGreedily can report (GPU_Null3.cell_records), so every
                 * random hit is found exactly and nothing is extrapolated. Chunks
                 * of sequences are halved if the buffer overflows. */
                const int64_t cap = 64000000LL;
                if (cells.size() < (size_t)cap * 4) cells.resize((size_t)cap * 4);
                /* The first chunk (16 sequences) measures cells per sequence; later
                 * chunks are sized to fill 85% of the buffer. */
                std::vector<std::pair<int, int> > todo;
                double per_seq = 0.0;
                int done_to = 0;
                while (done_to < nseq || !todo.empty()) {
                    if (todo.empty()) {
                        int n = per_seq > 0.0 ? (int)std::max(1.0, 0.85 * (double)cap / per_seq) : 16;
                        todo.push_back(std::make_pair(done_to, std::min(nseq, done_to + n)));
                        done_to = todo.back().second;
                    }
                    const int lo = todo.back().first, hi = todo.back().second;
                    todo.pop_back();
                    std::vector<Segment> segs;
                    for (int si = lo; si < hi; si++) segs.push_back({si, '+', 1, (int64_t)seqs[si].n});
                    int count = 0;
                    GPU_Null3 em = {};
                    em.enabled = use_n3 ? 1 : 0;
                    em.min_raw = -1e30f;
                    em.log2_omega = use_n3 ? log2f(cmm->null3_omega) : 0.0f;
                    for (int a = 0; a < 4; a++) em.log2_null[a] = log2f(cmm->null0[a]);
                    em.h_cells = cells.data(); em.h_cell_count = &count; em.cell_cap = (int)cap; em.cell_cut = -1e29f;
                    em.cell_records = 1; em.cell_margin = 0.05f;
                    std::vector<GPU_Window> cw; std::vector<WinMeta> cmeta; std::vector<float> cs, cb; int cstride = 0;
                    passes++;
                    double tq0 = now_sec();
                    if (run_stage(devm, kind, segs, cw, cmeta, cs, cb, cstride, 0.0f, NULL, &em) != 0) {
                        fprintf(stderr, "  [FATAL] CM %s %s: calibration scan failed\n", cm->name, mode_name[mode]);
                        return 1;
                    }
                    tg_exact += now_sec() - tq0;
                    tq0 = now_sec();
                    if (count > cap || count < 0) {
                        if (hi - lo == 1) { fprintf(stderr, "  [FATAL] CM %s %s: %d record cells exceed the buffer\n", cm->name, mode_name[mode], count); return 1; }
                        /* count is exact unless the int wrapped; re-split accordingly. */
                        per_seq = std::max(per_seq, count > 0 ? (double)count / (double)(hi - lo) : 2.0 * (double)cap / (double)(hi - lo));
                        const int n = std::min(hi - lo - 1, (int)std::max(1.0, 0.85 * (double)cap / per_seq));
                        for (int x = hi; x > lo; x -= n) todo.push_back({std::max(lo, x - n), x});
                        th_exact += now_sec() - tq0;
                        continue;
                    }
                    cells_exact += count;
                    per_seq = std::max(per_seq, (double)count / (double)(hi - lo));
                    /* Host side, all on worker threads: (1) bucket cell indices by
                     * sequence; (2) per sequence, counting sort by j and sort by d
                     * within j; ReportHitsGreedily per j; overlap removal. */
                    const int nsq = hi - lo;
                    const int nthr = std::max(1, std::min(32, (int)std::thread::hardware_concurrency()));
                    auto run_threads = [&](const std::function<void(int)> &fn) {
                        std::vector<std::thread> pool;
                        for (int t = 1; t < nthr; t++) pool.emplace_back(fn, t);
                        fn(0);
                        for (auto &th : pool) th.join();
                    };
                    auto cell_seq = [&](int c) -> int {   /* sequence slot, or -1 if outside the trusted range */
                        const int *cell = &cells[(size_t)c * 4];
                        const WinMeta &m = cmeta[cell[0]];
                        const int jp = cell[1] + 1, d = cell[2];
                        if (jp < m.trust_lo_jp || jp > m.win_len || d < rootlo || d > roothi) return -1;
                        return m.seq_idx - lo;
                    };
                    std::vector<std::vector<int> > hist(nthr, std::vector<int>(nsq + 1, 0));
                    const int64_t slice = ((int64_t)count + nthr - 1) / nthr;
                    run_threads([&](int t) {
                        const int64_t c0 = std::min<int64_t>(count, t * slice), c1 = std::min<int64_t>(count, c0 + slice);
                        for (int64_t c = c0; c < c1; c++) { int q = cell_seq((int)c); if (q >= 0) hist[t][q]++; }
                    });
                    std::vector<int64_t> seq_off(nsq + 1, 0);
                    std::vector<std::vector<int64_t> > place(nthr, std::vector<int64_t>(nsq, 0));
                    for (int q = 0; q < nsq; q++) {
                        int64_t run = seq_off[q];
                        for (int t = 0; t < nthr; t++) { place[t][q] = run; run += hist[t][q]; }
                        seq_off[q + 1] = run;
                    }
                    std::vector<int> idx((size_t)seq_off[nsq]);
                    run_threads([&](int t) {
                        const int64_t c0 = std::min<int64_t>(count, t * slice), c1 = std::min<int64_t>(count, c0 + slice);
                        for (int64_t c = c0; c < c1; c++) { int q = cell_seq((int)c); if (q >= 0) idx[(size_t)place[t][q]++] = (int)c; }
                    });
                    std::vector<std::vector<float> > got(nsq);
                    std::atomic<int> next(0);
                    run_threads([&](int) {
                        std::vector<CalCand> wc;
                        std::vector<char> wcov;
                        std::vector<CalCell> cc;
                        std::vector<int> jb;
                        for (int q; (q = next++) < nsq; ) {
                            const int si = lo + q;
                            const int L = seqs[si].n;
                            const int64_t n = seq_off[q + 1] - seq_off[q];
                            jb.assign((size_t)L + 2, 0);
                            cc.resize((size_t)n);
                            for (int64_t r = seq_off[q]; r < seq_off[q + 1]; r++) {
                                const int *cell = &cells[(size_t)idx[(size_t)r] * 4];
                                jb[(size_t)(cmeta[cell[0]].s + cell[1]) + 1]++;
                            }
                            for (int j = 1; j <= L + 1; j++) jb[j] += jb[j - 1];
                            for (int64_t r = seq_off[q]; r < seq_off[q + 1]; r++) {
                                const int *cell = &cells[(size_t)idx[(size_t)r] * 4];
                                const int j = (int)(cmeta[cell[0]].s + cell[1]);
                                float sc; memcpy(&sc, &cell[3], sizeof(float));
                                cc[(size_t)jb[j]++] = {si, j, cell[2], sc};
                            }
                            /* jb[j] now marks the end of j's cells; ascending d within j */
                            for (size_t k0 = 0, k = 0; k0 < cc.size(); k0 = k) {
                                for (k = k0 + 1; k < cc.size() && cc[k].j == cc[k0].j; k++) {}
                                if (k - k0 > 1) std::sort(cc.begin() + k0, cc.begin() + k,
                                                          [](const CalCell &x, const CalCell &y) { return x.d < y.d; });
                            }
                            const std::vector<double> &pre = calib_pre[si];
                            wc.clear();
                            size_t k = 0;
                            while (k < cc.size()) {
                                const int j = cc[k].j;
                                float max_rep = -1e30f;
                                for (; k < cc.size() && cc[k].j == j; k++) {
                                    const float raw = cc[k].sc; const int d = cc[k].d;
                                    if (!(raw > max_rep)) continue;
                                    float sc = raw;
                                    if (use_n3) sc -= calib_null3(&pre[(size_t)j * 4], &pre[(size_t)(j - d) * 4], d, cmm->null0, cmm->null3_omega);
                                    if (sc > max_rep) { wc.push_back({sc, j - d + 1, j}); max_rep = sc; }
                                }
                            }
                            calib_resolve(wc, L, wcov, got[q]);
                        }
                    });
                    for (auto &g : got) exact_scores.insert(exact_scores.end(), g.begin(), g.end());
                    th_exact += now_sec() - tq0;
                }
                P = nseq;
                exact_res = total_res;
                std::sort(exact_scores.begin(), exact_scores.end(), std::greater<float>());
                const long nrand = (long)exact_scores.size();
                const float cut = -1e29f;
                std::vector<float> all_tail = exact_scores;
                (void)tail_scores; (void)tg_tail; (void)th_tail; (void)cells_tail; (void)exact_res;
#else
                while (1) {
                    std::vector<Segment> segs;
                    exact_res = 0.0;
                    for (int si = 0; si < P; si++) { segs.push_back({si, '+', 1, (int64_t)seqs[si].n}); exact_res += seqs[si].n; }
                    int64_t cap = (int64_t)(2.2 * exact_res * band) + 1000000;
                    if (cap > 2000000000LL / 4) cap = 2000000000LL / 4;
                    if (cells.size() < (size_t)cap * 4) cells.resize((size_t)cap * 4);
                    int count = 0;
                    GPU_Null3 em = {};
                    em.h_cells = cells.data(); em.h_cell_count = &count; em.cell_cap = (int)cap; em.cell_cut = -1e29f;
                    std::vector<GPU_Window> cw; std::vector<WinMeta> cmeta; std::vector<float> cs, cb; int cstride = 0;
                    passes++;
                    double tq0 = now_sec();
                    if (run_stage(devm, kind, segs, cw, cmeta, cs, cb, cstride, 0.0f, NULL, &em) != 0) {
                        fprintf(stderr, "  [FATAL] CM %s %s: calibration scan failed\n", cm->name, mode_name[mode]);
                        return 1;
                    }
                    tg_exact += now_sec() - tq0;
                    tq0 = now_sec();
                    if (count > cap) {           /* too many cells: halve the exact set */
                        if (P == 1) { fprintf(stderr, "  [FATAL] CM %s %s: root band too wide for --calib-budget\n", cm->name, mode_name[mode]); return 1; }
                        P = std::max(1, P / 2);
                        continue;
                    }
                    cells_exact = count;
                    /* Dense root matrix per sequence, then Infernal's per-j reporting. */
                    std::vector<int64_t> row0(P + 1, 0);
                    for (int si = 0; si < P; si++) row0[si + 1] = row0[si] + seqs[si].n;
                    std::vector<float> mat((size_t)row0[P] * band, -1e30f);
                    for (int c = 0; c < count; c++) {
                        const int *cell = &cells[(size_t)c * 4];
                        const WinMeta &m = cmeta[cell[0]];
                        int jp = cell[1] + 1, d = cell[2];
                        if (jp < m.trust_lo_jp || jp > m.win_len || d < rootlo || d > roothi) continue;
                        int64_t j = m.s + jp - 1;
                        float sc; memcpy(&sc, &cell[3], sizeof(float));
                        mat[(size_t)(row0[m.seq_idx] + j - 1) * band + (d - rootlo)] = sc;
                    }
                    for (int si = 0; si < P; si++) {
                        const std::vector<double> &pre = calib_pre[si];
                        cand.clear();
                        for (int j = 1; j <= seqs[si].n; j++) {
                            const float *row = &mat[(size_t)(row0[si] + j - 1) * band];
                            float max_rep = -1e30f;
                            int dmax_j = std::min(roothi, j);
                            for (int d = rootlo; d <= dmax_j; d++) {
                                float raw = row[d - rootlo];
                                if (!(raw > max_rep)) continue;
                                float sc = raw;
                                if (use_n3) sc -= calib_null3(&pre[(size_t)j * 4], &pre[(size_t)(j - d) * 4], d, cmm->null0, cmm->null3_omega);
                                if (sc > max_rep) { cand.push_back({sc, j - d + 1, j}); max_rep = sc; }
                            }
                        }
                        calib_resolve(cand, seqs[si].n, covered, exact_scores);
                    }
                    th_exact += now_sec() - tq0;
                    break;
                }
                std::sort(exact_scores.begin(), exact_scores.end(), std::greater<float>());
                long nrand = (long)exact_scores.size();
                float cut = -1e29f;
                std::vector<float> all_tail = exact_scores;
                if (P < nseq) {
                    /* Remaining sequences: only cells above a raw-score cut. Every
                     * hit scoring >= cut is then exact (cells below it can neither
                     * be such a hit nor block one). Total hit count is extrapolated. */
                    nrand = (long)llround((double)exact_scores.size() * total_res / exact_res);
                    size_t r = (size_t)ceil(1.5 * nhits_to_fit * exact_res / total_res) + 5;
                    if (exact_scores.empty()) { fprintf(stderr, "  [FATAL] CM %s %s: no hits in the exact pass\n", cm->name, mode_name[mode]); return 1; }
                    cut = exact_scores[std::min(r, exact_scores.size() - 1)] - 1.5f;
                    /* Glocal scores are flat across (j,d): on large models tens of
                     * millions of cells clear the cut. Sequences are therefore
                     * scanned in chunks whose possible cells stay below 2^31 (the
                     * device counter is an int), and a chunk that overflows the
                     * buffer is halved; one sequence always fits (cap >= n*band). */
                    int64_t maxn = 1;
                    for (int si = P; si < nseq; si++) maxn = std::max<int64_t>(maxn, seqs[si].n);
                    const int64_t cap = std::max<int64_t>(16000000LL, maxn * band + 1000000LL);
                    const int per_chunk = (int)std::max<int64_t>(1, 1500000000LL / (maxn * band));
                    for (int attempt = 0; ; attempt++) {
                        if (cells.size() < (size_t)cap * 4) cells.resize((size_t)cap * 4);
                        std::vector<std::pair<int, int> > todo;
                        for (int lo = nseq; lo > P; lo -= per_chunk) todo.push_back({std::max(P, lo - per_chunk), lo});
                        tail_scores.clear();
                        cells_tail = 0;
                        while (!todo.empty()) {
                            const int lo = todo.back().first, hi = todo.back().second;
                            todo.pop_back();
                            std::vector<Segment> segs;
                            for (int si = lo; si < hi; si++) segs.push_back({si, '+', 1, (int64_t)seqs[si].n});
                            int count = 0;
                            GPU_Null3 em = {};
                            em.h_cells = cells.data(); em.h_cell_count = &count; em.cell_cap = (int)cap; em.cell_cut = cut;
                            std::vector<GPU_Window> cw; std::vector<WinMeta> cmeta; std::vector<float> cs, cb; int cstride = 0;
                            passes++;
                            double tq0 = now_sec();
                            if (run_stage(devm, kind, segs, cw, cmeta, cs, cb, cstride, 0.0f, NULL, &em) != 0) {
                                fprintf(stderr, "  [FATAL] CM %s %s: calibration scan failed\n", cm->name, mode_name[mode]);
                                return 1;
                            }
                            tg_tail += now_sec() - tq0;
                            tq0 = now_sec();
                            if (count > cap || count < 0) {
                                if (hi - lo == 1) { fprintf(stderr, "  [FATAL] CM %s %s: %d tail cells exceed the buffer\n", cm->name, mode_name[mode], count); return 1; }
                                const int mid = lo + (hi - lo) / 2;
                                todo.push_back({mid, hi});
                                todo.push_back({lo, mid});
                                th_tail += now_sec() - tq0;
                                continue;
                            }
                            cells_tail += count;
                            std::vector<CalCell> cc; cc.reserve(count);
                            for (int c = 0; c < count; c++) {
                                const int *cell = &cells[(size_t)c * 4];
                                const WinMeta &m = cmeta[cell[0]];
                                int jp = cell[1] + 1, d = cell[2];
                                if (jp < m.trust_lo_jp || jp > m.win_len || d < rootlo || d > roothi) continue;
                                float sc; memcpy(&sc, &cell[3], sizeof(float));
                                cc.push_back({m.seq_idx, (int)(m.s + jp - 1), d, sc});
                            }
                            std::sort(cc.begin(), cc.end(), [](const CalCell &a, const CalCell &b) {
                                if (a.seq != b.seq) return a.seq < b.seq;
                                if (a.j != b.j) return a.j < b.j;
                                return a.d < b.d;
                            });
                            size_t k = 0;
                            while (k < cc.size()) {
                                int si = cc[k].seq;
                                const std::vector<double> &pre = calib_pre[si];
                                cand.clear();
                                while (k < cc.size() && cc[k].seq == si) {
                                    int j = cc[k].j;
                                    float max_rep = -1e30f;
                                    for (; k < cc.size() && cc[k].seq == si && cc[k].j == j; k++) {
                                        float raw = cc[k].sc; int d = cc[k].d;
                                        if (!(raw > max_rep)) continue;
                                        float sc = raw;
                                        if (use_n3) sc -= calib_null3(&pre[(size_t)j * 4], &pre[(size_t)(j - d) * 4], d, cmm->null0, cmm->null3_omega);
                                        if (sc > max_rep) { cand.push_back({sc, j - d + 1, j}); max_rep = sc; }
                                    }
                                }
                                std::vector<float> got;
                                calib_resolve(cand, seqs[si].n, covered, got);
                                for (float s : got) if (s >= cut) tail_scores.push_back(s);
                            }
                            th_tail += now_sec() - tq0;
                        }
                        double tq0 = now_sec();
                        all_tail = exact_scores;
                        all_tail.insert(all_tail.end(), tail_scores.begin(), tail_scores.end());
                        std::sort(all_tail.begin(), all_tail.end(), std::greater<float>());
                        float tp = nhits_to_fit / (float)nrand;
                        long need = (long)((double)nrand * (double)tp);
                        th_tail += now_sec() - tq0;
                        if (need >= 2 && (long)all_tail.size() >= need && all_tail[need - 1] >= cut) break;
                        if (attempt >= 5) { fprintf(stderr, "  [FATAL] CM %s %s: tail not covered after lowering the cut\n", cm->name, mode_name[mode]); return 1; }
                        cut -= 2.0f;
                    }
                }
#endif
                /* cmcalibrate fit_histogram + esl_exp_FitComplete + SetExpInfo. */
                const float tailp = nhits_to_fit / (float)nrand;
                const long nfit = (long)((double)nrand * (double)tailp);
                if (tailp > 1.0f || nfit < 2 || (long)all_tail.size() < nfit) {
                    fprintf(stderr, "  [FATAL] CM %s %s: too few hits to fit (%ld hits, %ld needed)\n", cm->name, mode_name[mode], nrand, nfit);
                    return 1;
                }
                double mu = all_tail[nfit - 1], mean = 0.0;
                for (long q = 0; q < nfit; q++) mean += (double)all_tail[q] - mu;
                mean /= (double)nfit;
                double lambda = 1.0 / mean;
                double mu_extrap = mu - log(1.0 / (double)tailp) / lambda;
                fprintf(calib_fp, "%s\t%s\t%s\t%.5f\t%.5f\t%.5f\t%.0f\t%ld\t%f\t%ld\t%d\t%d\t%.3f\t%ld\t%ld\t%d\t%.3f\n",
                        cm->name, cm->acc, mode_name[mode], lambda, mu_extrap, mu, total_res, nrand, tailp, nfit,
                        P == nseq ? 1 : 0, P, cut < -1e28f ? 0.0f : cut, cells_exact, cells_tail, passes, now_sec() - tm0);
                fflush(calib_fp);
                if (getenv("GPU_PERF_CM_TIMING"))
                    fprintf(stderr, "[CALIB]\t%s\t%s\tM=%d\tW=%d\tband=%d\texact_seqs=%d\tgpu_exact=%.3f\thost_exact=%.3f\tgpu_tail=%.3f\thost_tail=%.3f\ttotal=%.3f\n",
                            cm->name, mode_name[mode], cm->M, W, band, P, tg_exact, th_exact, tg_tail, th_tail, now_sec() - tm0);
                if (!is_local) { gpu_cm_device_free(&devm); gpu_cm_host_destroy(cmm); }
            }
            total_gpu_ms += (now_sec() - t_up1) * 1000.0;
            gpu_cm_device_free(&dev);
            gpu_cm_host_destroy(cm);
            continue;
        }

        /* Per-endpoint candidate cutoff (raw Inside score at E = candidate_evalue
         * under the full search space). */
        const GPU_EvalueParams *evp_ins = &cm->evp_ins_local;
        double endpoint_cutoff = (double)min_score;
        if (!use_cyk && candidate_evalue > 0.0) {
            if (!evp_ins->is_valid || evp_ins->lambda <= 0.0f || evp_ins->dbsize <= 0.0 || evp_ins->nrandhits <= 0) {
                fprintf(stderr, "CM %s lacks valid calibration for --candidate-evalue\n", cm->name);
                return 1;
            }
            double ez = (2.0 * total_cm_count * (double)total_residues / evp_ins->dbsize) * evp_ins->nrandhits;
            if (ez > candidate_evalue)
                endpoint_cutoff = std::max(endpoint_cutoff,
                    (double)evp_ins->mu + (log(ez) - log(candidate_evalue)) / evp_ins->lambda);
        }
        /* --null3-gate: per-j gate scores become NULL3-corrected (Infernal
         * corrects every reported cell, CYK filter and final stage alike), and
         * the gate is lowered by the margin to absorb GPU vs Infernal rounding
         * of the raw score and of the correction itself. */
        GPU_Null3 n3_ins = {}, n3_cyk = {};
        /* Both gates carry the margin whenever a gate can drop a final hit:
         * the corrected gate (null3 may be ~0), and the cascade, which
         * otherwise leaves the Inside gate as the only margin-free step. */
        bool margin_gate = !use_cyk && candidate_evalue > 0.0 && (null3_gate || cascade_p > 0.0);
        if (margin_gate) endpoint_cutoff -= cyk_margin;
        if (null3_gate) {
            if (!cm->has_null3) { fprintf(stderr, "CM %s lacks NULL/N3OMEGA for --null3-gate\n", cm->name); return 1; }
            n3_ins.enabled = 1;
            n3_ins.min_raw = (float)endpoint_cutoff;
            n3_ins.log2_omega = log2f(cm->null3_omega);
            for (int a = 0; a < 4; a++) n3_ins.log2_null[a] = log2f(cm->null0[a]);
            n3_cyk = n3_ins;
        }

        std::vector<GPU_Window> all_windows;
        std::vector<WinMeta> all_meta;
        std::vector<float> all_scores, bestsc_j;
        int perj_stride = 0;
        double t0 = now_sec();
        int rc;
        if (use_cyk) {
            /* Legacy CYK keeps the per-window max (whole-tile gating); a
             * diagnostic path only. The cascade uses the fast per-j CYK. */
            rc = run_stage(dev, 2, *active_segs, all_windows, all_meta, all_scores, bestsc_j, perj_stride);
        } else if (cascade_p > 0.0) {
            /* GPU CYK -> Inside cascade mirroring Infernal's --nohmm pipeline
             * (pli_cyk_seq_filter then pli_final_stage).
             * Filter: beta1 bands, local CYK calibration (ECMLC), cutoff
             * mu - log(P)/lambda, lowered by --cyk-margin bits to absorb GPU vs
             * Infernal rounding. Raw CYK scores are used: any correction
             * Infernal applies only lowers scores, so this keeps a superset.
             * Infernal keeps [j-(W-1), i+(W-1)] for each reported CYK hit (i,j).
             * Every such hit is a root cell (j,d=j-i+1) above the cutoff, so
             * with d_min(j) = smallest passing length at j, the window
             * [j-W+1, j-d_min+W] contains all of them. Overlapping windows are
             * merged and the merged segments are rescanned with beta2 Inside. */
            const GPU_EvalueParams *ec = &cm->evp_cyk_local;
            if (!ec->is_valid || ec->lambda <= 0.0f) {
                fprintf(stderr, "CM %s lacks local CYK calibration (ECMLC) for --cascade-cyk-p\n", cm->name);
                return 1;
            }
            float cyk_cut = (float)((double)ec->mu - log(cascade_p) / (double)ec->lambda - cyk_margin);
            std::vector<GPU_Window> cw; std::vector<WinMeta> cm_meta; std::vector<float> cs, cb; int cstride = 0;
            std::vector<int> cdmin;
            double ta = now_sec();
            n3_cyk.min_raw = cyk_cut;
            rc = run_stage(dev_cyk, 1, *active_segs, cw, cm_meta, cs, cb, cstride, cyk_cut, &cdmin,
                           null3_gate ? &n3_cyk : NULL);
            double tb = now_sec();
            std::vector<Segment> surv;
            if (rc == 0) {
                for (size_t w = 0; w < cw.size(); w++) {
                    const WinMeta &m = cm_meta[w];
                    const float *bjw = cb.data() + w * cstride;
                    const int *dmw = cdmin.data() + w * cstride;
                    for (int jp = m.trust_lo_jp; jp <= m.win_len; jp++) {
                        if (!(bjw[jp - 1] >= cyk_cut)) continue;
                        int64_t j = m.s + jp - 1;
                        int dm = dmw[jp - 1];
                        if (dm < 1 || dm > W) dm = 1;   /* unknown: widest right bound */
                        surv.push_back({m.seq_idx, m.strand, std::max((int64_t)1, j - W + 1),
                                        std::min((int64_t)m.seq_len, j - dm + W)});
                    }
                }
                std::sort(surv.begin(), surv.end(), [](const Segment &x, const Segment &y) {
                    if (x.seq_idx != y.seq_idx) return x.seq_idx < y.seq_idx;
                    if (x.strand != y.strand) return x.strand < y.strand;
                    return x.lo < y.lo;
                });
                std::vector<Segment> merged;
                for (const Segment &g : surv) {
                    if (!merged.empty() && merged.back().seq_idx == g.seq_idx && merged.back().strand == g.strand
                        && g.lo <= merged.back().hi + 1)
                        merged.back().hi = std::max(merged.back().hi, g.hi);
                    else merged.push_back(g);
                }
                surv.swap(merged);
            }
            int64_t surv_nt = 0;
            for (const Segment &g : surv) surv_nt += g.hi - g.lo + 1;
            double tc = now_sec();
            if (rc == 0 && !surv.empty())
                rc = run_stage(dev, 0, surv, all_windows, all_meta, all_scores, bestsc_j, perj_stride,
                               0.0f, NULL, null3_gate ? &n3_ins : NULL);
            double td = now_sec();
            if (getenv("GPU_PERF_CM_TIMING"))
                fprintf(stderr, "[CASCADE]\t%s\tcyk_cut=%.4f\tcyk_tiles=%zu\tcyk_s=%.6f\tsegments=%zu\tsurvivor_nt=%ld\tfraction=%.6g\tinside_tiles=%zu\tinside_s=%.6f\n",
                        cm->name, cyk_cut, cw.size(), tb - ta, surv.size(), (long)surv_nt,
                        (double)surv_nt / ((top_only ? 1.0 : 2.0) * (double)scanned_residues), all_windows.size(), td - tc);
        } else {
            rc = run_stage(dev, 0, *active_segs, all_windows, all_meta, all_scores, bestsc_j, perj_stride,
                           0.0f, NULL, null3_gate ? &n3_ins : NULL);
        }
        int nw_total = (int)all_windows.size();
        double t1 = now_sec();
        total_gpu_ms += (t1 - t0) * 1000.0;
        if (getenv("GPU_PERF_CM_TIMING")) {
            fprintf(stderr, "[CM_PERF]\t%s\tM=%d\tW=%d\twindows=%d\tbatch=%d\tscan_s=%.9f\n",
                    cm->name, cm->M, cm->W, nw_total, ws_batch, t1 - t0);
        }

        if (rc != 0) {
            fprintf(stderr, "  [FATAL] CM %d (%s): batch scan failed (%d windows)\n",
                    n_cms, cm->name, nw_total);
            return 1;
        } else {
            const GPU_EvalueParams *evp = use_cyk ? &cm->evp_cyk_local : &cm->evp_ins_local;

            if (use_cyk) {
                /* Legacy CYK: one max score per whole tile (no per-j output). */
                for (int w = 0; w < nw_total; w++) {
                    if (all_scores[w] > min_score) {
                        const WinMeta &m = all_meta[w];
                        int64_t cs_i0 = m.s, cs_j0 = m.s + m.win_len - 1;
                        ScanHit hit;
                        hit.cm_name = cm->name; hit.cm_acc = cm->acc;
                        hit.seq_name = seqs[m.seq_idx].name;
                        hit.seq_idx = m.seq_idx; hit.strand = m.strand;
                        if (m.strand == '+') { hit.i0 = cs_i0; hit.j0 = cs_j0; }
                        else { hit.i0 = m.seq_len - cs_i0 + 1; hit.j0 = m.seq_len - cs_j0 + 1; }
                        hit.score = all_scores[w];
                        hit.gate_export = false;
                        hit.evalue = gpu_evalue_from_score(all_scores[w], evp,
                                                           total_residues, total_cm_count);
                        all_hits.push_back(hit);
                        n_hits++;
                    }
                }
            } else {
                /* Gate each endpoint before clustering (endpoint_cutoff above).
                 * A score>0 chain can span megabases and assign one peak score
                 * to many unrelated RNAs. null3 only lowers a hit's score, so a
                 * permissive raw E gate (or the corrected gate lowered by the
                 * margin) retains a candidate superset under the same
                 * calibration/Z. This emits candidate regions, NOT final hits. */
                /* Plan B Inside: collect candidate end-positions from each tile's
                 * trustworthy region, in cur_seq coordinates, keyed by
                 * (seq_idx, strand). Then merge nearby end-positions (gap ≤ W)
                 * into hit regions: one reported hit per region, score = peak
                 * bestsc, span = [first cand, last cand] of the region. */
                struct Cand { int seq_idx; char strand; int64_t j_cs; float sc; };
                std::vector<Cand> cands;
                for (int w = 0; w < nw_total; w++) {
                    const WinMeta &m = all_meta[w];
                    const float *bj = bestsc_j.data() + (size_t)w * perj_stride;
                    for (int jp = m.trust_lo_jp; jp <= m.win_len; jp++) {
                        float sc = bj[jp - 1];     /* jp0 = jp-1 (0-based) */
                        if (sc > endpoint_cutoff) {
                            Cand c;
                            c.seq_idx = m.seq_idx;
                            c.strand  = m.strand;
                            c.j_cs    = m.s + jp - 1;  /* cur_seq end coordinate */
                            c.sc      = sc;
                            cands.push_back(c);
                        }
                    }
                }
                /* Sort by (seq_idx, strand, j_cs) so a region is a contiguous run. */
                std::sort(cands.begin(), cands.end(), [](const Cand &a, const Cand &b) {
                    if (a.seq_idx != b.seq_idx) return a.seq_idx < b.seq_idx;
                    if (a.strand  != b.strand)  return a.strand  < b.strand;
                    return a.j_cs < b.j_cs;
                });
                int64_t merge_gap = W;   /* end-positions ≤ W apart belong together */
                size_t k = 0, N = cands.size();
                while (k < N) {
                    int seq_idx = cands[k].seq_idx;
                    char strand = cands[k].strand;
                    int64_t first_j = cands[k].j_cs, last_j = cands[k].j_cs;
                    float peak = cands[k].sc;
                    k++;
                    while (k < N && cands[k].seq_idx == seq_idx && cands[k].strand == strand
                           && cands[k].j_cs - last_j <= merge_gap) {
                        last_j = cands[k].j_cs;
                        if (cands[k].sc > peak) peak = cands[k].sc;
                        k++;
                    }
                    int64_t cs_i0 = first_j, cs_j0 = last_j;  /* region span in cur_seq */
                    /* Endpoint j may represent any hit length 1..W. Preserve
                     * that possible start for refinement; region span alone is
                     * not the sequence whose composition belongs to its score. */
                    if (candidate_evalue > 0.0) cs_i0 = std::max((int64_t)1, first_j - W + 1);
                    int seq_len = seqs[seq_idx].n;
                    ScanHit hit;
                    hit.cm_name = cm->name; hit.cm_acc = cm->acc;
                    hit.seq_name = seqs[seq_idx].name;
                    hit.seq_idx = seq_idx; hit.strand = strand;
                    if (strand == '+') {
                        hit.i0 = cs_i0; hit.j0 = cs_j0;
                    } else {
                        /* cur_seq is the reverse-complement; map span back to the
                         * forward-strand genome coordinate frame (same convention
                         * as the old code: pos → seq_len - pos + 1). */
                        hit.i0 = seq_len - cs_i0 + 1;
                        hit.j0 = seq_len - cs_j0 + 1;
                    }
                    hit.score  = peak;
                    hit.evalue = gpu_evalue_from_score(peak, evp, total_residues, total_cm_count);
                    hit.gate_export = margin_gate;
                    all_hits.push_back(hit);
                    n_hits++;
                }
            }
        }

        /* --hits-stream: publish this model's candidates as soon as they exist
         * (write to .tmp, then rename), so CPU refinement can overlap with the
         * remaining GPU scan. Models without exported candidates get no file. */
        if (hits_stream_dir && all_hits.size() > hits_mark) {
            char base[1024], tmp[1100];
            std::string nm = cm->name;
            for (auto &c : nm) if (c == '/' || c == ' ') c = '_';
            snprintf(base, sizeof(base), "%s/%06d_%s.tsv", hits_stream_dir, n_cms, nm.c_str());
            snprintf(tmp, sizeof(tmp), "%s.tmp", base);
            FILE *sfp = fopen(tmp, "w");
            if (!sfp) { fprintf(stderr, "  [FATAL] cannot write %s\n", tmp); return 1; }
            int nw = write_hits_tsv(sfp, all_hits.data() + hits_mark, all_hits.data() + all_hits.size(), evalue_cutoff);
            fclose(sfp);
            if (nw > 0) { if (rename(tmp, base) != 0) { fprintf(stderr, "  [FATAL] rename %s\n", tmp); return 1; } }
            else remove(tmp);
        }

        gpu_cm_device_free(&dev);
        gpu_cm_host_destroy(cm);
        if (cm_cyk) { gpu_cm_device_free(&dev_cyk); gpu_cm_host_destroy(cm_cyk); }

        if (n_cms % 100 == 0) {
            fprintf(stderr, "  ... processed %d CMs, %d hits so far (GPU: %.1f s)\n",
                    n_cms, n_hits, total_gpu_ms / 1000.0);
        }
    }

    fclose(cm_fp);

    double total_wall = now_sec() - total_wall_start;

    /* ── Summary ── */
    printf("\n%s\n", "════════════════════════════════════════════════════════════");
    printf("  CMSCAN BENCHMARK RESULTS\n");
    printf("%s\n", "════════════════════════════════════════════════════════════");
    printf("  CMs processed:       %d\n", n_cms);
    printf("  Hits (score > 0):    %d\n", n_hits);
    printf("  GPU scan time:       %.2f s  (%.1f ms/CM)\n",
           total_gpu_ms / 1000.0, n_cms > 0 ? total_gpu_ms / n_cms : 0.0);
    printf("  CM upload time:      %.2f s  (%.1f ms/CM)\n",
           total_upload_ms / 1000.0, n_cms > 0 ? total_upload_ms / n_cms : 0.0);
    printf("  Total wall time:     %.2f s\n", total_wall);
    printf("  Throughput:          %.1f CMs/sec\n",
           total_wall > 0 ? n_cms / total_wall : 0.0);

    /* All hits sorted by E-value, TSV format for downstream comparison */
    if (!all_hits.empty()) {
        std::sort(all_hits.begin(), all_hits.end(),
                  [](const ScanHit &a, const ScanHit &b) { return a.evalue < b.evalue; });

        printf("\n  Top 20 hits:\n");
        printf("  %-20s %-12s %8s %12s  %s\n", "CM_NAME", "ACCESSION", "SCORE", "E-VALUE", "SEQ");
        int n_show = std::min((int)all_hits.size(), 20);
        for (int i = 0; i < n_show; i++) {
            const ScanHit &h = all_hits[i];
            printf("  %-20s %-12s %8.2f %12.2e  seq_%d\n",
                   h.cm_name.c_str(), h.cm_acc.c_str(), h.score, h.evalue, h.seq_idx);
        }

        /* Write top 100 hits TSV to stderr for validation */
        int n_tsv = std::min((int)all_hits.size(), 100);
        fprintf(stderr, "\n#GPU_HITS_TSV_START\n");
        fprintf(stderr, "cm_name\tcm_acc\tseq_idx\ti0\tj0\tstrand\tscore\tevalue\n");
        for (int i = 0; i < n_tsv; i++) {
            const ScanHit &h = all_hits[i];
            fprintf(stderr, "%s\t%s\t%d\t%ld\t%ld\t%c\t%.4f\t%.6e\n",
                    h.cm_name.c_str(), h.cm_acc.c_str(), h.seq_idx,
                    (long)h.i0, (long)h.j0, h.strand, h.score, h.evalue);
        }
        fprintf(stderr, "#GPU_HITS_TSV_END\n");
    }

    /* ── Write hits to file for downstream traceback ── */
    if (hits_out_path) {
        FILE *hfp = fopen(hits_out_path, "w");
        if (hfp) {
            int n_written = write_hits_tsv(hfp, all_hits.data(), all_hits.data() + all_hits.size(), evalue_cutoff);
            fclose(hfp);
            printf("  Hits written to %s: %d entries\n", hits_out_path, n_written);
        } else {
            fprintf(stderr, "WARNING: cannot open %s for writing\n", hits_out_path);
        }
    }

    if (hits_stream_dir) {
        char done[1024];
        snprintf(done, sizeof(done), "%s/DONE", hits_stream_dir);
        FILE *dfp = fopen(done, "w");
        if (!dfp) { fprintf(stderr, "  [FATAL] cannot write %s\n", done); return 1; }
        fprintf(dfp, "models\t%d\n", n_cms);
        fclose(dfp);
    }

    printf("\n=== cmscan Benchmark Complete ===\n");
    return 0;
}
