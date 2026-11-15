/*
 * cm_file_parser.h — Parse Infernal 1.1.x text CM files into GPU_CM_Host.
 *
 * Header-only. Reads the ASCII "INFERNAL1/a" format, extracts state
 * topology, transition scores, and emission scores.
 *
 * Local mode: beginsc/endsc and init_sc are computed from PBEGIN/PEND
 * values and node topology, matching Infernal's cm_localize() + FCalcInitDPScores().
 *
 * Alphabet: Kp=18 (full RNA IUPAC, matching Infernal's Easel RNA alphabet).
 * Digital codes: A=0, C=1, G=2, U=3, gap=4, R=5..N=15, nonresidue=16, missing=17.
 * Degenerate emission scores are computed as the average over constituent
 * canonical bases, matching Infernal's FCalcOptimizedEmitScores() / esl_abc_FAvgScore().
 */

#ifndef CM_FILE_PARSER_H
#define CM_FILE_PARSER_H

#include "gpu_types.h"
#include "infernal_gpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static int cmparse_sttype(const char *s) {
    if (s[0] == 'S'  && s[1] == '\0') return GPU_S_st;
    if (s[0] == 'D'  && s[1] == '\0') return GPU_D_st;
    if (s[0] == 'B'  && s[1] == '\0') return GPU_B_st;
    if (s[0] == 'E'  && s[1] == '\0') return GPU_E_st;
    if (strcmp(s, "IL") == 0) return GPU_IL_st;
    if (strcmp(s, "IR") == 0) return GPU_IR_st;
    if (strcmp(s, "MP") == 0) return GPU_MP_st;
    if (strcmp(s, "ML") == 0) return GPU_ML_st;
    if (strcmp(s, "MR") == 0) return GPU_MR_st;
    return -1;
}

/*  stid = f(node_type, position_within_node)
 *  ROOT:  0,1,2   BEGL: 3   BEGR: 4,5   MATP: 6..11
 *  MATL: 12..14    MATR: 15..17   END: 18   BIF: 19       */
static int8_t cmparse_stid(const char *nd, int pos) {
    if (strcmp(nd, "ROOT") == 0)  return (int8_t)(0  + pos);
    if (strcmp(nd, "BEGL") == 0)  return 3;
    if (strcmp(nd, "BEGR") == 0)  return (int8_t)(4  + pos);
    if (strcmp(nd, "MATP") == 0)  return (int8_t)(6  + pos);
    if (strcmp(nd, "MATL") == 0)  return (int8_t)(12 + pos);
    if (strcmp(nd, "MATR") == 0)  return (int8_t)(15 + pos);
    if (strcmp(nd, "END")  == 0)  return 18;
    if (strcmp(nd, "BIF")  == 0)  return 19;
    return -1;
}

/* Infernal node type constants (for local begin/end computation) */
#define CMPARSE_ROOT_nd  0
#define CMPARSE_BEGL_nd  1
#define CMPARSE_BEGR_nd  2
#define CMPARSE_MATP_nd  3
#define CMPARSE_MATL_nd  4
#define CMPARSE_MATR_nd  5
#define CMPARSE_BIF_nd   6
#define CMPARSE_END_nd   7

static int cmparse_ndtype(const char *s) {
    if (strcmp(s, "ROOT") == 0) return CMPARSE_ROOT_nd;
    if (strcmp(s, "BEGL") == 0) return CMPARSE_BEGL_nd;
    if (strcmp(s, "BEGR") == 0) return CMPARSE_BEGR_nd;
    if (strcmp(s, "MATP") == 0) return CMPARSE_MATP_nd;
    if (strcmp(s, "MATL") == 0) return CMPARSE_MATL_nd;
    if (strcmp(s, "MATR") == 0) return CMPARSE_MATR_nd;
    if (strcmp(s, "BIF")  == 0) return CMPARSE_BIF_nd;
    if (strcmp(s, "END")  == 0) return CMPARSE_END_nd;
    return -1;
}

/* Parse a float value or '*' (IMPOSSIBLE) from a string, advancing the pointer. */
static float cmparse_score(char **pp) {
    char *p = *pp;
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '*') {
        p++;
        *pp = p;
        return GPU_IMPOSSIBLE;
    }
    float val;
    int consumed = 0;
    if (sscanf(p, "%f%n", &val, &consumed) == 1) {
        *pp = p + consumed;
        return val;
    }
    return GPU_IMPOSSIBLE;
}

static int cmparse_n_canonical_emissions(int st) {
    if (st == GPU_MP_st) return 16;
    if (st == GPU_ML_st || st == GPU_MR_st ||
        st == GPU_IL_st || st == GPU_IR_st) return 4;
    return 0;
}

static int cmparse_n_expanded_emissions(int st, int Kp) {
    if (st == GPU_MP_st) return Kp * Kp;
    if (st == GPU_ML_st || st == GPU_MR_st ||
        st == GPU_IL_st || st == GPU_IR_st) return Kp;
    return 0;
}

/*
 * RNA IUPAC degeneracy: bitmask of which canonical bases (A=0,C=1,G=2,U=3)
 * each digital code represents. Matches Easel RNA alphabet "ACGU-RYMKSWHBVDN*~".
 */
static const uint8_t RNA_DEGEN_MASK[18] = {
    0x1, 0x2, 0x4, 0x8,  /* 0-3: A, C, G, U */
    0x0,                  /* 4: gap */
    0x5, 0xA, 0x3, 0xC,  /* 5-8: R(AG), Y(CU), M(AC), K(GU) */
    0x6, 0x9,             /* 9-10: S(CG), W(AU) */
    0xB, 0xE, 0x7, 0xD,  /* 11-14: H(!G), B(!A), V(!U), D(!C) */
    0xF,                  /* 15: N(any) */
    0x0, 0x0,             /* 16-17: nonresidue, missing */
};

static int cmparse_popcount4(uint8_t m) {
    return (m & 1) + ((m >> 1) & 1) + ((m >> 2) & 1) + ((m >> 3) & 1);
}

/*
 * Expand 4 canonical singlet emission scores to Kp=18.
 * Degenerate codes get the average score over their constituent bases,
 * matching Infernal's esl_abc_FAvgScore().
 */
static void cmparse_expand_singlet(const float *canon4, float *out18) {
    for (int a = 0; a < 4; a++) out18[a] = canon4[a];
    for (int a = 4; a < 18; a++) {
        uint8_t mask = RNA_DEGEN_MASK[a];
        int cnt = cmparse_popcount4(mask);
        if (cnt == 0) { out18[a] = GPU_IMPOSSIBLE; continue; }
        float sum = 0.0f;
        for (int b = 0; b < 4; b++)
            if (mask & (1 << b)) sum += canon4[b];
        out18[a] = sum / cnt;
    }
}

/*
 * Expand 4x4 canonical pair emission scores to 18x18.
 * canon16 layout: [left_canon * 4 + right_canon] (K=4 stride)
 * out324 layout:  [left_code * 18 + right_code]  (Kp=18 stride)
 */
static void cmparse_expand_pair(const float *canon16, float *out324) {
    for (int i = 0; i < 324; i++) out324[i] = GPU_IMPOSSIBLE;
    for (int a = 0; a < 18; a++) {
        uint8_t ma = RNA_DEGEN_MASK[a];
        int ca = cmparse_popcount4(ma);
        if (ca == 0) continue;
        for (int b = 0; b < 18; b++) {
            uint8_t mb = RNA_DEGEN_MASK[b];
            int cb = cmparse_popcount4(mb);
            if (cb == 0) continue;
            float sum = 0.0f;
            for (int ai = 0; ai < 4; ai++) {
                if (!(ma & (1 << ai))) continue;
                for (int bi = 0; bi < 4; bi++) {
                    if (!(mb & (1 << bi))) continue;
                    sum += canon16[ai * 4 + bi];
                }
            }
            out324[a * 18 + b] = sum / (ca * cb);
        }
    }
}

/*
 * parse_cm_file — read an INFERNAL1/a text .cm file, return GPU_CM_Host*.
 *
 * If global_mode is true (or parse_cm_file is called), uses global-mode scoring.
 * If parse_cm_file_local is called, computes local begin/end scores matching
 * Infernal's cm_localize() + CMLogoddsify() + FCalcInitDPScores().
 *
 * For multi-CM files (e.g., Rfam.cm), use parse_cm_from_fp() with an open FILE*.
 * It reads one CM and returns, leaving the FILE* positioned at the next CM.
 */
static GPU_CM_Host *parse_cm_from_fp(FILE *fp, int local_mode);
/* qdb_bands: 1 = beta1 (tight, Infernal's CYK filter), 2 = beta2 (loose,
 * final Inside stage), 0 = legacy environment choice (GPU_QDB_TIGHT). */
static GPU_CM_Host *parse_cm_from_fp_bands(FILE *fp, int local_mode, int qdb_bands);
static GPU_CM_Host *parse_cm_file_mode(const char *path, int local_mode);

static GPU_CM_Host *parse_cm_file(const char *path) {
    return parse_cm_file_mode(path, 0);
}

static GPU_CM_Host *parse_cm_file_local(const char *path) {
    return parse_cm_file_mode(path, 1);
}

static GPU_CM_Host *
parse_cm_file_mode(const char *path, int local_mode)
{
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "cm_parser: cannot open %s\n", path); return NULL; }
    GPU_CM_Host *cm = parse_cm_from_fp(fp, local_mode);
    fclose(fp);
    return cm;
}

static GPU_CM_Host *
parse_cm_from_fp(FILE *fp, int local_mode)
{
    return parse_cm_from_fp_bands(fp, local_mode, 0);
}

static GPU_CM_Host *
parse_cm_from_fp_bands(FILE *fp, int local_mode, int qdb_bands)
{
    static int cm_parse_count = 0;
    cm_parse_count++;

    char line[8192];
    int M = 0, W = 0, NODES = 0;
    float el_selfsc = 0.0f;
    float null0[4] = {0.25f, 0.25f, 0.25f, 0.25f};
    float null3_omega = 0.0f;
    int   n3_seen = 0;
    float pbegin = 0.5f, pend = 0.5f;

    char cm_name[GPU_CM_NAME_MAX] = "";
    char cm_acc[GPU_CM_NAME_MAX]  = "";

    /* E-value calibration */
    GPU_EvalueParams evp_cyk_local  = {0.0f, 0.0f, 0.0, 0, 0};
    GPU_EvalueParams evp_ins_local  = {0.0f, 0.0f, 0.0, 0, 0};

    /* ── Seek to INFERNAL1/a header if not at one ── */
    int found_header = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "INFERNAL1/a", 11) == 0) { found_header = 1; break; }
    }
    if (!found_header) return NULL;

    /* ── Parse header ── */
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "NAME", 4) == 0 && (line[4] == ' ' || line[4] == '\t')) {
            char buf[GPU_CM_NAME_MAX];
            if (sscanf(line + 4, " %255s", buf) == 1)
                strncpy(cm_name, buf, GPU_CM_NAME_MAX - 1);
        }
        else if (strncmp(line, "ACC", 3) == 0 && (line[3] == ' ' || line[3] == '\t')) {
            char buf[GPU_CM_NAME_MAX];
            if (sscanf(line + 3, " %255s", buf) == 1)
                strncpy(cm_acc, buf, GPU_CM_NAME_MAX - 1);
        }
        else if (strncmp(line, "STATES", 6) == 0)  sscanf(line + 6, "%d", &M);
        else if (strncmp(line, "NODES", 5) == 0)  sscanf(line + 5, "%d", &NODES);
        else if (line[0] == 'W' && line[1] == ' ') sscanf(line + 1, "%d", &W);
        else if (strncmp(line, "ELSELF", 6) == 0) sscanf(line + 6, "%f", &el_selfsc);
        else if (strncmp(line, "N3OMEGA", 7) == 0) { if (sscanf(line + 7, "%f", &null3_omega) == 1) n3_seen |= 1; }
        else if (strncmp(line, "NULL ", 5) == 0) {
            /* prob2ascii: log2(p / (1/K)); '*' would mean p = 0 */
            float lv[4];
            if (sscanf(line + 5, "%f %f %f %f", &lv[0], &lv[1], &lv[2], &lv[3]) == 4) {
                for (int a = 0; a < 4; a++) null0[a] = 0.25f * powf(2.0f, lv[a]);
                n3_seen |= 2;
            }
        }
        else if (strncmp(line, "PBEGIN", 6) == 0) sscanf(line + 6, "%f", &pbegin);
        else if (strncmp(line, "PEND", 4) == 0 && (line[4] == ' ' || line[4] == '\t'))
            sscanf(line + 4, "%f", &pend);
        else if (strncmp(line, "ECMLC", 5) == 0) {
            /* ECMLC format: lambda  mu_extrap  mu_orig  dbsize  nrandhits  tailp */
            float mu_orig_unused;
            long dbsize_long; int nrh;
            sscanf(line + 5, " %f %f %f %ld %d",
                   &evp_cyk_local.lambda, &evp_cyk_local.mu, &mu_orig_unused,
                   &dbsize_long, &nrh);
            evp_cyk_local.dbsize = (double)dbsize_long;
            evp_cyk_local.nrandhits = nrh;
            evp_cyk_local.is_valid = 1;
        }
        else if (strncmp(line, "ECMLI", 5) == 0) {
            /* ECMLI format: lambda  mu_extrap  mu_orig  dbsize  nrandhits  tailp */
            float mu_orig_unused;
            long dbsize_long; int nrh;
            sscanf(line + 5, " %f %f %f %ld %d",
                   &evp_ins_local.lambda, &evp_ins_local.mu, &mu_orig_unused,
                   &dbsize_long, &nrh);
            evp_ins_local.dbsize = (double)dbsize_long;
            evp_ins_local.nrandhits = nrh;
            evp_ins_local.is_valid = 1;
        }
        else if (line[0] == 'C' && line[1] == 'M' &&
                 (line[2] == '\n' || line[2] == '\r' || line[2] == '\0' || line[2] == ' '))
            break;
    }
    if (M == 0 || W == 0) {
        fprintf(stderr, "cm_parser: failed to read STATES/W from header\n");
        return NULL;
    }

    int Kp = 18;
    int W1 = W + 1;

    /* ── Temporary arrays ── */
    int8_t *sttype = (int8_t *)calloc(M, sizeof(int8_t));
    int8_t *stid   = (int8_t *)calloc(M, sizeof(int8_t));
    int    *cfirst = (int *)calloc(M, sizeof(int));
    int    *cnum   = (int *)calloc(M, sizeof(int));

    float  *tsc = (float *)malloc((size_t)M * GPU_MAXCONNECT * sizeof(float));
    for (int i = 0; i < M * GPU_MAXCONNECT; i++) tsc[i] = GPU_IMPOSSIBLE;

    int oesc_cap = M * 20;
    float *oesc = (float *)malloc(oesc_cap * sizeof(float));
    int   *oesc_offset = (int *)malloc(M * sizeof(int));
    for (int i = 0; i < M; i++) oesc_offset[i] = -1;
    int oesc_total = 0;

    int *dmin1_arr = (int *)calloc(M, sizeof(int));
    int *dmax1_arr = (int *)malloc(M * sizeof(int));
    for (int i = 0; i < M; i++) dmax1_arr[i] = W;

    /* Node topology for local mode */
    int max_nodes = (NODES > 0) ? NODES : M;
    int *ndtype  = (int *)calloc(max_nodes, sizeof(int));
    int *nodemap = (int *)calloc(max_nodes, sizeof(int));
    int  n_nodes = 0;

    char cur_nd[16] = "";
    int  cur_nd_idx = -1;
    int  pos_in_nd  = 0;
    int  states_read = 0;

    /* ── Parse CM body ── */
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '/' && line[1] == '/') break;

        /* Node annotation line: e.g. "                         [ MATP  1 ]  ..." */
        if (strchr(line, '[')) {
            char *b = strchr(line, '[');
            int nd_idx = -1;
            sscanf(b + 1, " %15s %d", cur_nd, &nd_idx);
            pos_in_nd = 0;
            if (nd_idx >= 0 && nd_idx < max_nodes) {
                ndtype[nd_idx]  = cmparse_ndtype(cur_nd);
                nodemap[nd_idx] = -1;   /* will be set when first state is parsed */
                cur_nd_idx = nd_idx;
                if (nd_idx >= n_nodes) n_nodes = nd_idx + 1;
            }
            continue;
        }

        /* Try to parse as state line */
        char st_str[8];
        int v, pl, pn, cf, cn, dm2, dm1, dx1, dx2;
        int nr;
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '\0' || *p == '\n' || *p == '\r') continue;

        if (sscanf(p, "%7s %d %d %d %d %d %d %d %d %d%n",
                   st_str, &v, &pl, &pn, &cf, &cn,
                   &dm2, &dm1, &dx1, &dx2, &nr) < 10)
            continue;

        int st = cmparse_sttype(st_str);
        if (st < 0 || v < 0 || v >= M) continue;

        sttype[v] = (int8_t)st;
        stid[v]   = cmparse_stid(cur_nd, pos_in_nd);
        cfirst[v] = cf;
        cnum[v]   = cn;
        /* QDB band columns on a state line are: dmin2 dmin1 dmax1 dmax2.
         * Infernal's MAIN search
         * (cmsearch/cmscan final stage, cm_pipeline.c:3681) AND cmcalibrate
         * (cmcalibrate.c:2338) both scan with SMX_QDB2_LOOSE = the WIDER beta2
         * (1e-15) bands [dmin2, dmax2]. We therefore DEFAULT to beta2 to match
         * deployed search. Only the CYK filter and Infernal's `benchmark-search`
         * driver use the tighter beta1 (1e-7) [dmin1, dmax1]; set GPU_QDB_TIGHT=1
         * to select beta1 when validating against benchmark-search.
         * GPU_QDB_LOOSE is still honored (forces beta2) for backward compat. */
        int qdb_tight = qdb_bands ? (qdb_bands == 1) : (getenv("GPU_QDB_TIGHT") != NULL);
        if (qdb_tight) {
            dmin1_arr[v] = dm1;
            dmax1_arr[v] = (dx1 > W) ? W : dx1;
        } else {
            dmin1_arr[v] = dm2;
            dmax1_arr[v] = (dx2 > W) ? W : dx2;   /* alpha sized for W; scan caps at W anyway */
        }

        if (cur_nd_idx >= 0 && cur_nd_idx < max_nodes && pos_in_nd == 0)
            nodemap[cur_nd_idx] = v;
        pos_in_nd++;

        p += nr;

        /* Transition scores */
        int n_tsc = (st == GPU_B_st || st == GPU_E_st) ? 0 : cn;
        for (int t = 0; t < n_tsc && t < GPU_MAXCONNECT; t++)
            tsc[v * GPU_MAXCONNECT + t] = cmparse_score(&p);

        /* Emission scores: read canonical, expand to Kp=18 */
        int n_canon = cmparse_n_canonical_emissions(st);
        int n_expanded = cmparse_n_expanded_emissions(st, Kp);
        if (n_canon > 0) {
            float canon_buf[16];
            for (int e = 0; e < n_canon; e++)
                canon_buf[e] = cmparse_score(&p);

            while (oesc_total + n_expanded > oesc_cap) {
                oesc_cap *= 2;
                oesc = (float *)realloc(oesc, oesc_cap * sizeof(float));
            }
            oesc_offset[v] = oesc_total;
            if (st == GPU_MP_st)
                cmparse_expand_pair(canon_buf, oesc + oesc_total);
            else
                cmparse_expand_singlet(canon_buf, oesc + oesc_total);
            oesc_total += n_expanded;
        }
        states_read++;
    }
    if (states_read != M)
        fprintf(stderr, "cm_parser: expected %d states, read %d\n", M, states_read);

    /* ── Compute beginsc, endsc, init_sc ── */
    float *beginsc = (float *)malloc(M * sizeof(float));
    float *endsc   = (float *)malloc(M * sizeof(float));
    float *init_sc = (float *)malloc((size_t)M * W1 * sizeof(float));
    for (int v = 0; v < M; v++) { beginsc[v] = GPU_IMPOSSIBLE; endsc[v] = GPU_IMPOSSIBLE; }
    for (int i = 0; i < M * W1; i++) init_sc[i] = GPU_IMPOSSIBLE;

    int has_local_begins = 0;

    if (local_mode && n_nodes >= 2) {
        has_local_begins = 1;

        /* ── Local begins (matching cm_CalculateLocalBeginProbs) ── */
        int nstarts = 0;
        for (int nd = 2; nd < n_nodes; nd++) {
            int ndt = ndtype[nd];
            if (ndt == CMPARSE_MATP_nd || ndt == CMPARSE_MATL_nd ||
                ndt == CMPARSE_MATR_nd || ndt == CMPARSE_BIF_nd)
                nstarts++;
        }
        if (nstarts > 0) {
            float bsc_node1 = logf(1.0f - pbegin) / logf(2.0f);
            float bsc_internal = logf(pbegin / (float)nstarts) / logf(2.0f);

            if (nodemap[1] >= 0 && nodemap[1] < M)
                beginsc[nodemap[1]] = bsc_node1;

            for (int nd = 2; nd < n_nodes; nd++) {
                int ndt = ndtype[nd];
                if ((ndt == CMPARSE_MATP_nd || ndt == CMPARSE_MATL_nd ||
                     ndt == CMPARSE_MATR_nd || ndt == CMPARSE_BIF_nd) &&
                    nodemap[nd] >= 0 && nodemap[nd] < M)
                    beginsc[nodemap[nd]] = bsc_internal;
            }
        }

        /* Zero out ROOT (v=0) transition scores — in local mode,
         * all entry is through beginsc, ROOT transitions are IMPOSSIBLE */
        for (int t = 0; t < GPU_MAXCONNECT; t++)
            tsc[0 * GPU_MAXCONNECT + t] = GPU_IMPOSSIBLE;

        /* ── Local ends (matching cm_localize) ── */
        int nexits = 0;
        for (int nd = 1; nd < n_nodes; nd++) {
            int ndt = ndtype[nd];
            int ndt_next = (nd + 1 < n_nodes) ? ndtype[nd + 1] : CMPARSE_END_nd;
            if ((ndt == CMPARSE_MATP_nd || ndt == CMPARSE_MATL_nd ||
                 ndt == CMPARSE_MATR_nd || ndt == CMPARSE_BEGL_nd ||
                 ndt == CMPARSE_BEGR_nd) &&
                ndt_next != CMPARSE_END_nd)
                nexits++;
        }

        if (nexits > 0) {
            float end_prob = pend / (float)nexits;
            float esc_local = logf(end_prob) / logf(2.0f);
            float tsc_adjust = logf(1.0f / (1.0f + end_prob)) / logf(2.0f);

            for (int nd = 1; nd < n_nodes; nd++) {
                int ndt = ndtype[nd];
                int ndt_next = (nd + 1 < n_nodes) ? ndtype[nd + 1] : CMPARSE_END_nd;
                if ((ndt == CMPARSE_MATP_nd || ndt == CMPARSE_MATL_nd ||
                     ndt == CMPARSE_MATR_nd || ndt == CMPARSE_BEGL_nd ||
                     ndt == CMPARSE_BEGR_nd) &&
                    ndt_next != CMPARSE_END_nd) {
                    int v = nodemap[nd];
                    if (v >= 0 && v < M) {
                        endsc[v] = esc_local;
                        int cn_v = cnum[v];
                        for (int t = 0; t < cn_v && t < GPU_MAXCONNECT; t++)
                            tsc[v * GPU_MAXCONNECT + t] += tsc_adjust;
                    }
                }
            }
        }

        /* ── init_sc = FCalcInitDPScores: el_selfsc * d + endsc[v] ── */
        for (int v = 0; v < M; v++) {
            if (GPU_NOT_IMPOSSIBLE(endsc[v])) {
                for (int d = 0; d <= W; d++)
                    init_sc[v * W1 + d] = el_selfsc * d + endsc[v];
            }
        }

        if (cm_parse_count <= 5 || cm_parse_count % 500 == 0)
            fprintf(stderr, "cm_parser: local mode — nstarts=%d nexits=%d pbegin=%.2f pend=%.2f\n",
                    nstarts, nexits, pbegin, pend);
    }

    /* ── Create host CM ── */
    GPU_CM_Host *cm = gpu_cm_host_create(
        M, W, Kp, el_selfsc, has_local_begins,
        sttype, stid, cfirst, cnum,
        tsc, oesc, oesc_offset, oesc_total,
        beginsc, endsc, init_sc,
        dmin1_arr, dmax1_arr);

    free(sttype); free(stid); free(cfirst); free(cnum);
    free(tsc); free(oesc); free(oesc_offset);
    free(beginsc); free(endsc); free(init_sc);
    free(dmin1_arr); free(dmax1_arr);
    free(ndtype); free(nodemap);

    /* Store E-value calibration and identity */
    cm->evp_cyk_local = evp_cyk_local;
    cm->evp_ins_local = evp_ins_local;
    {
        float z = null0[0] + null0[1] + null0[2] + null0[3];
        for (int a = 0; a < 4; a++) cm->null0[a] = null0[a] / z;   /* Infernal renormalizes */
    }
    cm->null3_omega = null3_omega;
    cm->has_null3 = (n3_seen == 3 && null3_omega > 0.0f);
    strncpy(cm->name, cm_name, GPU_CM_NAME_MAX - 1);
    strncpy(cm->acc, cm_acc, GPU_CM_NAME_MAX - 1);

    if (cm_parse_count <= 5 || cm_parse_count % 500 == 0)
        fprintf(stderr, "cm_parser: loaded %s [%s] — M=%d  W=%d  n_begl=%d  oesc_total=%d  local=%d  evp_ins=%d  (#%d)\n",
                cm->name, cm->acc, cm->M, cm->W, cm->n_begl, cm->oesc_total,
                has_local_begins, evp_ins_local.is_valid, cm_parse_count);
    return cm;
}

#endif /* CM_FILE_PARSER_H */
