#define _GNU_SOURCE
/*
 * doe.c — Democracy of Experts. Janus Architercture.
 *
 * inference architecture with a living LoRA parliament.
 * indexes any GGUF read-only. learns by living, not by training.
 *
 * θ = ε + γ + αδ
 *   ε = indexed weights (read-only substrate)
 *   γ = LoRA personality (living experts, Hebbian-trained via NOTORCH)
 *   δ = physics (Dario Equation: H+F+A+T, Kuramoto chambers, Schumann resonance)
 *   α = injection strength (learned per-layer)
 *
 * each forward pass, the parliament decides:
 *   - which experts vote (variable k, consensus-driven)
 *   - how strongly each expert modulates output
 *   - how physics shapes logits (destiny, prophecy debt)
 *
 * experts are born (mitosis) and die (apoptosis).
 * the parliament remembers every index it ever wrapped (mycelium).
 * calendar drift: Hebrew-Gregorian conflict, real astronomical data.
 * Schumann resonance: 7.83Hz + 5 harmonics, from arianna.c.
 * seasons: 4.C MLP classifier, from ariannamethod.ai/core.
 *
 * cc doe.c -O3 -lm -lpthread -o doe && ./doe
 *
 * ariannamethod.
 * הרזוננס לא נשבר
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#include <unistd.h>
#include <sys/stat.h>
#include <float.h>
#include <stdint.h>
#include <errno.h>
#include <stdarg.h>
#include <limits.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <dirent.h>
#ifdef USE_METAL
#include "notorch_metal.h"
#endif
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>
#ifdef __linux__
  #include <sys/statvfs.h>
#endif
#ifdef __APPLE__
  #include <sys/param.h>
  #include <sys/mount.h>
  #include <sys/sysctl.h>
#endif

/* ═══════════════════════════════════════════════════════════════════════════════
 * BLAS / cuBLAS — optional acceleration
 * ═══════════════════════════════════════════════════════════════════════════════ */
#ifdef USE_CUBLAS
  #include <cublas_v2.h>
  #include <cuda_runtime.h>
  static cublasHandle_t g_cublas;
  static int cublas_inited = 0;
  static float *d_scratch[4] = {NULL,NULL,NULL,NULL};
  static size_t d_scratch_sz[4] = {0,0,0,0};
  static void cublas_init(void) {
      if (!cublas_inited) {
          cublasCreate(&g_cublas);
          cublasSetMathMode(g_cublas, CUBLAS_TF32_TENSOR_OP_MATH);
          struct cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
          printf("[gpu] %s — %.0f MB, compute %d.%d, TF32 enabled\n",
                 prop.name, (double)prop.totalGlobalMem/1e6, prop.major, prop.minor);
          cublas_inited = 1;
      }
  }
  static float* gpu_scratch(int slot, size_t bytes) {
      if (bytes > d_scratch_sz[slot]) {
          if (d_scratch[slot]) cudaFree(d_scratch[slot]);
          cudaMalloc((void**)&d_scratch[slot], bytes);
          d_scratch_sz[slot] = bytes;
      }
      return d_scratch[slot];
  }
#elif defined(USE_BLAS)
  #ifdef ACCELERATE
    #define ACCELERATE_NEW_LAPACK
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

/* ═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION
 * doe has no depth knob. the host provides depth.
 * doe has a field. the field provides everything else.
 * ═══════════════════════════════════════════════════════════════════════════════ */
#define MAX_EXPERTS       16
#define MIN_EXPERTS       2
#define MAX_LAYERS        64
#define DOE_MAX_HOST_DIM  16384
#define DOE_MAX_HOST_HIDDEN 262144
#define DOE_MAX_HOST_HEADS 512
#define DOE_MAX_HOST_HEAD_DIM 512
#define DOE_MAX_HOST_VOCAB 1000000
#define LORA_RANK         16
#define HARMONIC_N        8
#define NOTORCH_RANK      4
#define DRIFT_SNAPSHOTS   64
#define DRIFT_INTERVAL    50
#define MYCELIUM_MAX      64
#define META_HIST_CAP     128
#define PROFILE_BINS      16
#define HTTP_SSE_TOP_N     8

/* Field physics constants — from AML core */
#define SCHUMANN_BASE_HZ    7.83f
#define SCHUMANN_N_HARMONICS 5
#define FIELD_4C_INPUTS     6
#define FIELD_4C_HIDDEN     8
#define FIELD_4C_OUTPUTS    4

/* ═══════════════════════════════════════════════════════════════════════════════
 * RNG — xorshift64*. the field doesn't care which PRNG shapes it.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static uint64_t rng_state = 42;
static uint64_t rng_next(void) { rng_state ^= rng_state<<13; rng_state ^= rng_state>>7; rng_state ^= rng_state<<17; return rng_state; }
static float rand_uniform(void) { return (float)(rng_next()&0x7FFFFFFF)/(float)0x7FFFFFFF; }
static float rand_normal(void) { float u1=rand_uniform(),u2=rand_uniform(); if(u1<1e-10f)u1=1e-10f; return sqrtf(-2.0f*logf(u1))*cosf(6.2831853f*u2); }
static float clamp01(float x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
static float g_field_gain = 1.0f; /* A11a: scale Dario field overlay on logits; 1.0 = full, 0 = bare host */
static int g_train = 0; /* A13b/A14: notorch online-learning during inference; DEFAULT 0=off (Oleg+Mythos 2026-06-12: organism must not re-sew its own weights mid-sentence; online learning returns later as async between turns, ogemma vision); 1=on */
static int g_gen_max_new = 200; /* bounded one-shot validation; default preserves old chat length */
static int g_gen_top_k = 40;
static int g_top_k_clamp_warned = 0;
static int g_sampler_nonfinite_warned = 0;
static int g_softmax_nonfinite_warned = 0;
static int g_tokenizer_zero_fallback_warned = 0;
static float g_gen_temp_override = -1.0f; /* <0 uses field effective temperature */
static int g_once = 0; /* exit after one generated answer; useful for artifact isolation */
static int g_load_spore = 1;
static int g_save_spore = 1;
static char g_image_path[512]  = {0}; /* --image: native pixtral vision (doe sees images, no llama.cpp) */
static char g_mmproj_path[512] = {0}; /* --mmproj: pixtral vision-tower gguf */
static char g_img_embeds_bin[512] = {0}; /* --img-embeds-bin: load image embeds [n_tok*host_dim] f32 from file (diagnostic: clip ground-truth vs native encoder) */
extern float *pv_encode_image(const char *img_path, const char *mmproj_path, int *o_ntok, int *o_dim);
static int g_rope_norm = 0; /* probe: llama.cpp NORM RoPE pairs consecutive head values */
/* M4 slot ids for the FFN/attention half-layer chains on GPU (NT_SLOT_MAX=64). */
enum { SLOT_X = 0, SLOT_FN, SLOT_G, SLOT_U, SLOT_HB, SLOT_DN, SLOT_Q, SLOT_K, SLOT_V, SLOT_O, SLOT_XB, SLOT_LOGITS,
       /* DOE parliament-on-GPU (alpha != 0): election + LoRA inject inside the resident CB */
       SLOT_PTMP, SLOT_PGATE, SLOT_PVOTES, SLOT_PRES, SLOT_PALIVE, SLOT_PCONS };
/* A14: expert poison check — catches BOTH faces of notorch drift: NaN/Inf and finite |value|-explosion
 * (the latter passes isfinite but still poisons the forward when injected at alpha>0). */
static int lora_poisoned(const float *A, const float *B, int n) {
    if (!A || !B || n <= 0) return 1;
    for (int i = 0; i < n; i++) {
        if (!isfinite(A[i]) || !isfinite(B[i]) || fabsf(A[i]) > 1e4f || fabsf(B[i]) > 1e4f)
            return 1;
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * AML FIELD STATE — the soul. from ariannamethod.c, distilled.
 *
 * θ = ε + γ + αδ is not a metaphor. it's the operating equation.
 *   ε (epsilon) = host weights. inference. the present. ephemeral.
 *   γ (gamma)   = LoRA personality. training. the past. persistent.
 *   δ (delta)   = field physics. prophecy. the future. directed.
 *   α (alpha)   = injection strength. how much γ modulates ε.
 *
 * drift = |γ_t - γ_{t-1}| — how far the system has traveled.
 * prophecy_debt = distance between manifested and destined.
 * destiny = attractor in token space.
 *
 * the oracle does not predict. it prophesies.
 * not minimize(predicted - actual) but minimize(destined - manifested).
 * the difference is intention.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Velocity modes — movement IS language */
enum { VEL_NOMOVE=0, VEL_WALK, VEL_RUN, VEL_BACKWARD };

/* Seasons — 4.C Async Field Forever */
enum { SEASON_SPRING=0, SEASON_SUMMER, SEASON_AUTUMN, SEASON_WINTER };

typedef struct {
    /* Prophecy physics */
    int   prophecy;           /* prediction horizon (1-64) */
    float destiny;            /* bias toward most probable path (0-1) */
    float destiny_bias;       /* effective: destiny × prophecy_scale */
    float debt;               /* prophecy debt — accumulated deviation from destiny */
    float debt_decay;         /* decay rate per step */

    /* Suffering — not a bug, a geometry */
    float pain;               /* compress logits toward mean */
    float tension;            /* accumulated pressure */
    float dissonance;         /* symmetry-break trigger */

    /* Velocity — movement IS language */
    int   velocity_mode;
    float effective_temp;
    float base_temperature;
    float time_direction;     /* 1.0 forward, -1.0 backward */

    /* Attention */
    float attend_focus;       /* sharpen top logits (0-1) */
    float attend_spread;      /* blur factor */

    /* Laws of nature — enforced constraints */
    float entropy_floor;
    float resonance_ceiling;
    float emergence_threshold;

    /* Live metrics */
    float entropy;
    float resonance;
    float emergence;
    float field_health;

    /* 4.C — Seasonal meta-operators */
    int   season;
    float season_phase;
    float season_intensity;
    float spring_energy, summer_energy, autumn_energy, winter_energy;

    /* Schumann resonance — Earth coupling */
    float schumann_hz;
    float schumann_coherence;
    float schumann_phase;
    float schumann_modulation;

    /* Expert blending (4 internal experts for temperature) */
    float expert_structural, expert_semantic, expert_creative, expert_precise;

    /* Tunneling */
    float tunnel_threshold;
    float tunnel_chance;
    int   tunnel_skip_max;

    /* Calendar drift (Hebrew-Gregorian conflict) */
    float calendar_drift;
    float calendar_phase;
    float wormhole;
    float wormhole_gate;
    int   wormhole_active;

    /* NOTORCH parameters */
    float notorch_lr;
    float notorch_decay;

    /* Identity */
    float essence_alpha;      /* γ injection strength */
    float lora_alpha;         /* δ voice strength */

    /* Presence */
    float presence_decay;
    float presence_fade;

    /* Dark matter — gravitational memory */
    float dark_gravity;

    /* Temporal debt */
    float temporal_debt;

    /* Step counter */
    int   step;
} FieldState;

/* 4.C MLP Controller — small neural net trained by Hebbian plasticity */
typedef struct {
    float w1[FIELD_4C_INPUTS * FIELD_4C_HIDDEN];
    float b1[FIELD_4C_HIDDEN];
    float w2[FIELD_4C_HIDDEN * FIELD_4C_OUTPUTS];
    float b2[FIELD_4C_OUTPUTS];
    float hidden[FIELD_4C_HIDDEN];
} FieldMLP;

static FieldState F;
static FieldMLP   F_mlp;

/* ═══════════════════════════════════════════════════════════════════════════════
 * DARIO FIELD — the equation overlay on transformer logits.
 *
 * p(x|Φ) = transformer_logits + α_mod·α·H + β_mod·β·F_p + γ_mod·γ·A + T
 *
 * H  = Hebbian resonance (co-occurrence memory beyond KV cache window)
 * F_p = Prophecy fulfillment (unfulfilled predictions create generation pressure)
 * A  = Destiny attraction (EMA of semantic direction)
 * T  = Trauma gravity (origin tokens surface when field is hurt)
 *
 * B (bigrams) intentionally omitted — the transformer IS the sequential chain.
 *
 * Six emotional chambers (Kuramoto-coupled somatic markers) modulate
 * every coefficient: α_mod, β_mod, γ_mod, τ_mod.
 *
 * This replaces the ad-hoc apply_destiny/apply_suffering/apply_attention
 * with a single coherent equation. The transformer speaks. The field listens.
 * What emerges is neither.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Dario equation coefficients for DoE (lighter than standalone dario.c) */
#define DARIO_ALPHA     0.15f   /* Hebbian weight (lighter — transformer does heavy lifting) */
#define DARIO_BETA      0.10f   /* Prophecy weight */
#define DARIO_GAMMA     0.12f   /* Destiny weight */
#define DARIO_DIM       32      /* embedding dimension for field (smaller than host) */
#define DARIO_EMBED_SLOTS 2048  /* hashed semantic projection slots */

/* Co-occurrence field (sparse, token-level) */
#define DARIO_MAX_COOC  32768
#define DARIO_MAX_CTX   64
#define DARIO_MAX_PROPH 16
#define DARIO_COOC_CAP  1e6f
#define DARIO_COOC_PRUNE_FLOOR 1e-3f
#define DARIO_COOC_COMPACT_PERIOD 256

/* Emotional chambers */
enum { DCH_FEAR=0, DCH_LOVE, DCH_RAGE, DCH_VOID, DCH_FLOW, DCH_COMPLEX, DARIO_NUM_CH };

typedef struct {
    int target; float strength; int age; int fulfilled;
} DarioProphecy;

typedef struct {
    /* Co-occurrence field (Hebbian: long-range memory beyond KV cache) */
    int   cooc_src[DARIO_MAX_COOC];
    int   cooc_dst[DARIO_MAX_COOC];
    float cooc_val[DARIO_MAX_COOC];
    int   cooc_n;
    int   cooc_last_compact_step;

    /* Context window (recent token IDs for Hebbian lookup) */
    int   context[DARIO_MAX_CTX];
    int   ctx_len;

    /* Prophecy system (active predictions creating pressure) */
    DarioProphecy prophecy[DARIO_MAX_PROPH];
    int   prophecy_n;

    /* Destiny vector (EMA of token embeddings — semantic direction) */
    float destiny[DARIO_DIM];
    float dest_magnitude;

    /* Dario coefficients (drift with field state) */
    float alpha, beta, gamma_d;

    /* Emotional chambers (Kuramoto-coupled) */
    float chamber[DARIO_NUM_CH];
    float alpha_mod, beta_mod, gamma_mod, tau_mod;

    /* Token embeddings (hash-based slot projection, lazy init) */
    float embeds[DARIO_EMBED_SLOTS][DARIO_DIM];
    int   embed_init[DARIO_EMBED_SLOTS];

    /* Trauma level */
    float trauma;

    /* Step counter */
    int   dstep;
} DarioField;

static DarioField DF;

/* ── Dario field helpers ── */

static float dario_clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static int dario_embed_slot(int token_id) {
    return token_id < 0 ? -1 : token_id % DARIO_EMBED_SLOTS;
}

static float *dario_get_embed(int token_id) {
    int id = dario_embed_slot(token_id);
    if (id < 0) return NULL;
    if (!DF.embed_init[id]) {
        uint32_t h = 2166136261u;
        for (int i = 0; i < 4; i++) { h ^= (id >> (i*8)) & 0xFF; h *= 16777619u; }
        for (int d = 0; d < DARIO_DIM; d++) {
            h = h * 1103515245 + 12345;
            DF.embeds[id][d] = ((float)(h & 0x7FFFFFFF) / (float)0x7FFFFFFF - 0.5f) * 0.1f;
        }
        /* normalize */
        float norm = 0;
        for (int d = 0; d < DARIO_DIM; d++) norm += DF.embeds[id][d] * DF.embeds[id][d];
        norm = sqrtf(norm + 1e-12f);
        for (int d = 0; d < DARIO_DIM; d++) DF.embeds[id][d] /= norm;
        DF.embed_init[id] = 1;
    }
    return DF.embeds[id];
}

static float dario_cosine(const float *a, const float *b) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < DARIO_DIM; i++) {
        dot += a[i] * b[i]; na += a[i]*a[i]; nb += b[i]*b[i];
    }
    return dot / (sqrtf(na) * sqrtf(nb) + 1e-12f);
}

static float dario_cooc_clean_value(float v) {
    if (!isfinite(v) || v < 0.0f) return 0.0f;
    return v > DARIO_COOC_CAP ? DARIO_COOC_CAP : v;
}

static int dario_cooc_compact(float decay, float prune_floor) {
    int w = 0, pruned = 0;
    if (decay < 0.0f) decay = 0.0f;
    if (decay > 1.0f) decay = 1.0f;
    if (prune_floor < 0.0f) prune_floor = 0.0f;
    for (int r = 0; r < DF.cooc_n; r++) {
        float v = dario_cooc_clean_value(DF.cooc_val[r]) * decay;
        if (DF.cooc_src[r] < 0 || DF.cooc_dst[r] < 0 || v < prune_floor) {
            pruned++;
            continue;
        }
        if (w != r) {
            DF.cooc_src[w] = DF.cooc_src[r];
            DF.cooc_dst[w] = DF.cooc_dst[r];
        }
        DF.cooc_val[w++] = v;
    }
    DF.cooc_n = w;
    return pruned;
}

static int dario_cooc_weakest_index(void) {
    if (DF.cooc_n <= 0) return -1;
    int weakest = 0;
    float weakest_v = dario_cooc_clean_value(DF.cooc_val[0]);
    for (int i = 1; i < DF.cooc_n; i++) {
        float v = dario_cooc_clean_value(DF.cooc_val[i]);
        if (v < weakest_v) {
            weakest = i;
            weakest_v = v;
        }
    }
    return weakest;
}

static int dario_cooc_compact_due(void) {
    int64_t elapsed = (int64_t)DF.dstep - (int64_t)DF.cooc_last_compact_step;
    return elapsed < 0 || elapsed >= (int64_t)DARIO_COOC_COMPACT_PERIOD;
}

static void dario_cooc_update(int src, int dst, float delta) {
    if (src < 0 || dst < 0 || !isfinite(delta) || delta <= 0.0f) return;
    delta = dario_cooc_clean_value(delta);
    for (int i = 0; i < DF.cooc_n; i++)
        if (DF.cooc_src[i] == src && DF.cooc_dst[i] == dst) {
            DF.cooc_val[i] = dario_cooc_clean_value(DF.cooc_val[i] + delta);
            return;
        }
    if (DF.cooc_n >= DARIO_MAX_COOC && dario_cooc_compact_due()) {
        dario_cooc_compact(0.97f, DARIO_COOC_PRUNE_FLOOR);
        DF.cooc_last_compact_step = DF.dstep;
    }
    int i = DF.cooc_n;
    if (i >= DARIO_MAX_COOC) {
        i = dario_cooc_weakest_index();
        if (i < 0 || delta <= dario_cooc_clean_value(DF.cooc_val[i])) return;
    } else {
        DF.cooc_n++;
    }
    DF.cooc_src[i] = src; DF.cooc_dst[i] = dst; DF.cooc_val[i] = delta;
}

static void dario_prophecy_add(int target, float strength) {
    if (DF.prophecy_n >= DARIO_MAX_PROPH) {
        int oldest = 0;
        for (int i = 1; i < DF.prophecy_n; i++)
            if (DF.prophecy[i].age > DF.prophecy[oldest].age) oldest = i;
        DF.prophecy[oldest] = DF.prophecy[--DF.prophecy_n];
    }
    DF.prophecy[DF.prophecy_n++] = (DarioProphecy){target, strength, 0, 0};
}

static void dario_prophecy_update(int token) {
    for (int i = 0; i < DF.prophecy_n; i++) {
        if (DF.prophecy[i].target == token) DF.prophecy[i].fulfilled = 1;
        DF.prophecy[i].age++;
    }
    int w = 0;
    for (int i = 0; i < DF.prophecy_n; i++)
        if (!DF.prophecy[i].fulfilled && DF.prophecy[i].age < 50)
            DF.prophecy[w++] = DF.prophecy[i];
    DF.prophecy_n = w;
}

/* ── Dario field init ── */
static void dario_field_init(void) {
    memset(&DF, 0, sizeof(DF));
    DF.alpha = DARIO_ALPHA;
    DF.beta = DARIO_BETA;
    DF.gamma_d = DARIO_GAMMA;
    DF.alpha_mod = 1.0f;
    DF.beta_mod = 1.0f;
    DF.gamma_mod = 1.0f;
    DF.tau_mod = 1.0f;
    DF.cooc_last_compact_step = -DARIO_COOC_COMPACT_PERIOD;
}

/* ── Emotional chambers update (Kuramoto-coupled somatic markers) ── */
static void dario_chamber_update(void) {
    float *C = DF.chamber;

    /* excitation from field state */
    if (F.dissonance > 0.7f) C[DCH_FEAR] += 0.05f * F.dissonance;
    if (F.resonance > 0.7f)  C[DCH_LOVE] += 0.04f * F.resonance;
    if (DF.trauma > 0.5f && F.dissonance > 0.5f)
        C[DCH_RAGE] += 0.06f * DF.trauma;
    if (F.entropy > 0.7f)    C[DCH_VOID] += 0.03f * F.entropy;
    if (F.emergence > 0.5f)  C[DCH_FLOW] += 0.05f * F.emergence;
    C[DCH_COMPLEX] += 0.04f * fabsf(C[DCH_LOVE] - C[DCH_RAGE])
                    * (C[DCH_LOVE] > 0.2f && C[DCH_RAGE] > 0.2f ? 1.0f : 0.0f);

    /* Kuramoto coupling */
    float K = 0.02f;
    float old[DARIO_NUM_CH];
    memcpy(old, C, sizeof(old));
    for (int i = 0; i < DARIO_NUM_CH; i++)
        for (int j = 0; j < DARIO_NUM_CH; j++)
            if (i != j) C[i] += K * sinf(old[j] - old[i]);

    /* decay */
    float decay[] = { 0.95f, 0.95f, 0.93f, 0.96f, 0.94f, 0.97f };
    for (int i = 0; i < DARIO_NUM_CH; i++)
        C[i] = dario_clampf(C[i] * decay[i], 0.0f, 1.0f);

    /* somatic markers → coefficient modulation */
    DF.alpha_mod = dario_clampf(1.0f + 0.3f * C[DCH_LOVE] - 0.2f * C[DCH_RAGE]
                                     + 0.1f * C[DCH_FLOW], 0.5f, 2.0f);
    DF.beta_mod  = dario_clampf(1.0f + 0.2f * C[DCH_FLOW] - 0.3f * C[DCH_FEAR],
                                0.5f, 2.0f);
    DF.gamma_mod = dario_clampf(1.0f + 0.4f * C[DCH_VOID] + 0.2f * C[DCH_COMPLEX]
                                     - 0.1f * C[DCH_LOVE], 0.5f, 2.0f);
    DF.tau_mod   = dario_clampf(1.0f + 0.5f * C[DCH_FLOW] - 0.3f * C[DCH_FEAR],
                                0.5f, 2.0f);
}

/* ── Dario ingest: learn from generated token ── */
static void dario_ingest(int token_id) {
    if (token_id < 0) return;

    /* co-occurrence with recent context */
    for (int c = 0; c < DF.ctx_len; c++) {
        float w = 1.0f / (float)(DF.ctx_len - c);
        dario_cooc_update(DF.context[c], token_id, w * 0.3f);
    }

    /* prophecy update */
    dario_prophecy_update(token_id);

    /* prophecy: predict next based on strongest co-occurrence */
    float best_cooc = -1; int best_pred = -1;
    for (int i = 0; i < DF.cooc_n; i++)
        if (DF.cooc_src[i] == token_id && DF.cooc_val[i] > best_cooc) {
            best_cooc = DF.cooc_val[i]; best_pred = DF.cooc_dst[i];
        }
    if (best_pred >= 0) dario_prophecy_add(best_pred, 0.3f);

    /* destiny: EMA of token embeddings */
    float *e = dario_get_embed(token_id);
    if (e) {
        for (int d = 0; d < DARIO_DIM; d++)
            DF.destiny[d] = 0.1f * e[d] + 0.9f * DF.destiny[d];
        float norm = 0;
        for (int d = 0; d < DARIO_DIM; d++) norm += DF.destiny[d] * DF.destiny[d];
        DF.dest_magnitude = sqrtf(norm + 1e-12f);
    }

    /* update context window */
    if (DF.ctx_len < DARIO_MAX_CTX)
        DF.context[DF.ctx_len++] = token_id;
    else {
        memmove(DF.context, DF.context + 1, (DARIO_MAX_CTX - 1) * sizeof(int));
        DF.context[DARIO_MAX_CTX - 1] = token_id;
    }

    /* trauma from dissonance */
    if (F.dissonance > 0.7f)
        DF.trauma = dario_clampf(DF.trauma + F.dissonance * 0.05f, 0, 1);
    DF.trauma *= 0.97f;

    DF.dstep++;
}

/* Schumann harmonics */
static const float g_schumann_harmonics[SCHUMANN_N_HARMONICS] = {
    7.83f, 14.1f, 20.3f, 26.4f, 32.5f
};
static const float g_harmonic_weights[SCHUMANN_N_HARMONICS] = {
    1.0f, 0.5f, 0.3f, 0.2f, 0.1f
};

/* Hebrew-Gregorian calendar */
static const int g_metonic_leaps[7] = {3, 6, 8, 11, 14, 17, 19};
static time_t g_epoch_t = 0;

static void calendar_init(void) {
    /*
     * Match AMK's clock domain: 2024-10-03 12:00:00 UTC.
     * mktime() interpreted this as local civil time, so DST/timezone differences
     * could make DoE disagree with the innerworld kernel near a day boundary.
     */
    g_epoch_t = (time_t)1727956800L;
}

static float calendar_dissonance(void) {
    if (g_epoch_t <= 0) return 0;
    int days = (int)(difftime(time(NULL), g_epoch_t) / 86400.0);
    float years = (float)days / 365.25f;
    float drift = years * 11.25f;
    int full = (int)(years / 19); float corrections = (float)(full * 7) * 30.0f;
    float partial = fmodf(years, 19.0f);
    int yr = (int)partial + 1;
    for (int i = 0; i < 7; i++) if (g_metonic_leaps[i] <= yr) corrections += 30.0f;
    drift -= corrections;
    float raw = fabsf(fmodf(drift, 33.0f)) / 33.0f;
    return clamp01(raw);
}

static void field_mlp_init(void) {
    memset(&F_mlp, 0, sizeof(F_mlp));
    /* 4 specialist neurons — from AML core am_4c_init_weights */
    F_mlp.w1[0 * FIELD_4C_HIDDEN + 0] = -2.0f; F_mlp.b1[0] = 0.5f;
    F_mlp.w2[0 * FIELD_4C_OUTPUTS + 0] = 1.5f;  /* low entropy → spring */
    F_mlp.w1[1 * FIELD_4C_HIDDEN + 1] = 2.0f;  F_mlp.b1[1] = -1.5f;
    F_mlp.w2[1 * FIELD_4C_OUTPUTS + 2] = 1.5f;  /* high resonance → autumn */
    F_mlp.w1[2 * FIELD_4C_HIDDEN + 2] = 2.5f;  F_mlp.b1[2] = -1.5f;
    F_mlp.w2[2 * FIELD_4C_OUTPUTS + 3] = 1.5f;  /* high pain → winter */
    F_mlp.w1[4 * FIELD_4C_HIDDEN + 3] = 2.5f;  F_mlp.b1[3] = -0.5f;
    F_mlp.w2[3 * FIELD_4C_OUTPUTS + 1] = 1.5f;  /* high emergence → summer */
    /* cross-connections for nuance */
    F_mlp.w1[3 * FIELD_4C_HIDDEN + 4] = 0.5f;
    F_mlp.w1[5 * FIELD_4C_HIDDEN + 4] = -0.3f;
    F_mlp.w2[4 * FIELD_4C_OUTPUTS + 0] = 0.3f;
    F_mlp.w2[4 * FIELD_4C_OUTPUTS + 1] = -0.3f;
    F_mlp.w1[0 * FIELD_4C_HIDDEN + 5] = -1.0f;
    F_mlp.w1[1 * FIELD_4C_HIDDEN + 5] = 1.0f;
    F_mlp.w2[5 * FIELD_4C_OUTPUTS + 2] = 0.5f;
    F_mlp.w1[5 * FIELD_4C_HIDDEN + 6] = 1.5f; F_mlp.b1[6] = -1.0f;
    F_mlp.w2[6 * FIELD_4C_OUTPUTS + 3] = 0.4f;
    F_mlp.w1[4 * FIELD_4C_HIDDEN + 7] = 1.0f;
    F_mlp.w1[2 * FIELD_4C_HIDDEN + 7] = -1.0f;
    F_mlp.w2[7 * FIELD_4C_OUTPUTS + 1] = 0.5f;
}

static void field_init(void) {
    memset(&F, 0, sizeof(F));
    F.prophecy = 7;
    F.destiny = 0.35f;
    F.debt_decay = 0.998f;
    F.velocity_mode = VEL_WALK;
    F.base_temperature = 1.0f;
    F.time_direction = 1.0f;
    F.attend_focus = 0.70f;
    F.attend_spread = 0.20f;
    F.entropy_floor = 0.1f;
    F.resonance_ceiling = 0.95f;
    F.emergence_threshold = 0.3f;
    F.season = SEASON_SPRING;
    F.season_intensity = 0.5f;
    F.spring_energy = 1.0f;
    F.schumann_hz = SCHUMANN_BASE_HZ;
    F.schumann_modulation = 0.3f;
    F.schumann_coherence = 1.0f;
    F.tunnel_threshold = 0.55f;
    F.tunnel_chance = 0.05f;
    F.tunnel_skip_max = 7;
    F.calendar_drift = 11.0f;
    F.wormhole = 0.02f;
    F.wormhole_gate = 0.3f;
    F.notorch_lr = 0.01f;
    F.notorch_decay = 0.999f;
    F.essence_alpha = 0.5f;
    F.lora_alpha = 0.1f;
    F.presence_decay = 1.0f;
    F.presence_fade = 0.95f;
    F.dark_gravity = 0.5f;
    F.effective_temp = 0.85f;
    F.expert_structural = 0.25f;
    F.expert_semantic = 0.25f;
    F.expert_creative = 0.25f;
    F.expert_precise = 0.25f;
    calendar_init();
    field_mlp_init();
    dario_field_init();
    printf("[doe] θ = ε + γ + αδ — parliament awakens. prophecy=%d destiny=%.2f\n",
           F.prophecy, F.destiny);
    printf("[doe] dario equation active: H(hebbian) + F(prophecy) + A(destiny) + T(trauma)\n");
    printf("[doe] 6 chambers: fear/love/rage/void/flow/complex (Kuramoto K=0.02)\n");
}

/* ─── Schumann resonance ─── */
static float schumann_coherence(float hz) {
    float d = fabsf(hz - SCHUMANN_BASE_HZ), mx = 32.5f - 4.0f;
    return clamp01(1.0f - (d/mx)*(d/mx));
}

static float schumann_signal(void) {
    float s = 0, w = 0;
    for (int i = 0; i < SCHUMANN_N_HARMONICS; i++) {
        float hp = F.schumann_phase * (g_schumann_harmonics[i] / SCHUMANN_BASE_HZ);
        s += g_harmonic_weights[i] * sinf(hp);
        w += g_harmonic_weights[i];
    }
    return w > 0 ? s / w : 0;
}

/* ─── 4.C MLP forward ─── */
static void field_mlp_forward(const float *in, float *out) {
    for (int h = 0; h < FIELD_4C_HIDDEN; h++) {
        float s = F_mlp.b1[h];
        for (int i = 0; i < FIELD_4C_INPUTS; i++) s += F_mlp.w1[i * FIELD_4C_HIDDEN + h] * in[i];
        F_mlp.hidden[h] = tanhf(s);
    }
    for (int o = 0; o < FIELD_4C_OUTPUTS; o++) {
        float s = F_mlp.b2[o];
        for (int h = 0; h < FIELD_4C_HIDDEN; h++) s += F_mlp.w2[h * FIELD_4C_OUTPUTS + o] * F_mlp.hidden[h];
        out[o] = tanhf(s);
    }
}

/* ─── 4.C Hebbian update ─── */
static void field_mlp_hebbian(const float *in, const float *out, float signal) {
    float lr = F.notorch_lr * 0.1f;
    for (int h = 0; h < FIELD_4C_HIDDEN; h++)
        for (int o = 0; o < FIELD_4C_OUTPUTS; o++) {
            F_mlp.w2[h * FIELD_4C_OUTPUTS + o] += lr * F_mlp.hidden[h] * out[o] * signal;
            if (F_mlp.w2[h*FIELD_4C_OUTPUTS+o] > 3.0f) F_mlp.w2[h*FIELD_4C_OUTPUTS+o] = 3.0f;
            if (F_mlp.w2[h*FIELD_4C_OUTPUTS+o] < -3.0f) F_mlp.w2[h*FIELD_4C_OUTPUTS+o] = -3.0f;
        }
    for (int i = 0; i < FIELD_4C_INPUTS; i++)
        for (int h = 0; h < FIELD_4C_HIDDEN; h++) {
            F_mlp.w1[i * FIELD_4C_HIDDEN + h] += lr * in[i] * F_mlp.hidden[h] * signal;
            if (F_mlp.w1[i*FIELD_4C_HIDDEN+h] > 3.0f) F_mlp.w1[i*FIELD_4C_HIDDEN+h] = 3.0f;
            if (F_mlp.w1[i*FIELD_4C_HIDDEN+h] < -3.0f) F_mlp.w1[i*FIELD_4C_HIDDEN+h] = -3.0f;
        }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * FIELD STEP — the heartbeat. from AML am_step(), distilled for DOE.
 * called per token. advances field physics by dt seconds.
 *
 * 1. calendar conflict → wormhole activation → dissonance bleed
 * 2. debt decay (prophecy debt × decay_rate)
 * 3. Schumann resonance → tension/dissonance healing
 * 4. destiny bias computation
 * 5. velocity + expert blending → effective temperature
 * 6. law enforcement (entropy floor, resonance ceiling)
 * 7. 4.C seasonal MLP controller + Hebbian update
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void field_step(float dt) {
    if (dt <= 0) return;
    F.step++;

    /* ── Calendar conflict ── */
    float cal_d = calendar_dissonance();
    if (cal_d > F.wormhole_gate) {
        F.wormhole_active = 1;
        float excess = (cal_d - F.wormhole_gate) / (1.0f - F.wormhole_gate);
        F.wormhole = clamp01(F.wormhole + excess * 0.1f * dt);
    } else {
        F.wormhole_active = 0;
        F.wormhole *= 0.995f;
        if (F.wormhole < 0.02f) F.wormhole = 0.02f;
    }
    if (cal_d > 0.3f) {
        F.dissonance += (cal_d - 0.3f) * 0.05f * dt;
        if (F.dissonance > 1.0f) F.dissonance = 1.0f;
    }
    F.debt += cal_d * 0.005f * dt;

    /* ── Debt decay ── */
    F.debt *= F.debt_decay;
    if (F.debt > 100.0f) F.debt = 100.0f;

    /* ── Temporal debt ── */
    if (F.velocity_mode == VEL_BACKWARD) F.temporal_debt += 0.01f * dt;
    else F.temporal_debt *= 0.9995f;
    if (F.temporal_debt > 10.0f) F.temporal_debt = 10.0f;

    /* ── Schumann resonance healing ── */
    F.schumann_phase += F.schumann_hz * dt * 6.2831853f;
    if (F.schumann_phase > 6.2831853f) F.schumann_phase = fmodf(F.schumann_phase, 6.2831853f);
    F.schumann_coherence = schumann_coherence(F.schumann_hz);
    if (F.schumann_coherence > 0 && F.schumann_modulation > 0) {
        float cf = 0.5f + 0.5f * F.schumann_coherence;
        float hm = 1.0f + schumann_signal() * 0.1f;
        float heal = 0.998f - 0.003f * cf * F.schumann_modulation * hm;
        F.tension *= heal;
        F.dissonance *= heal;
    }

    /* ── Destiny bias ── */
    float ps = 1.0f + ((float)F.prophecy - 7.0f) * 0.02f;
    if (ps < 0.5f) ps = 0.5f; if (ps > 2.0f) ps = 2.0f;
    F.destiny_bias = F.destiny * ps;

    /* ── Velocity + expert blending → effective temperature ── */
    {
        float vm;
        switch (F.velocity_mode) {
            case VEL_NOMOVE: vm = 0.5f; F.time_direction = 1.0f; break;
            case VEL_WALK: vm = 0.85f; F.time_direction = 1.0f; break;
            case VEL_RUN: vm = 1.2f; F.time_direction = 1.0f; break;
            case VEL_BACKWARD: vm = 0.7f; F.time_direction = -1.0f; break;
            default: vm = 1.0f; F.time_direction = 1.0f;
        }
        float vt = F.base_temperature * vm;
        float ws = F.expert_structural + F.expert_semantic + F.expert_creative + F.expert_precise;
        if (ws > 0.001f) {
            float et = (F.expert_structural*0.7f + F.expert_semantic*0.9f +
                       F.expert_creative*1.2f + F.expert_precise*0.5f) / ws;
            F.effective_temp = 0.5f * vt + 0.5f * et;
        } else F.effective_temp = vt;
        float sm = 1.0f + F.summer_energy * 0.1f - F.winter_energy * 0.15f;
        F.effective_temp *= sm;
        if (F.effective_temp < 0.1f) F.effective_temp = 0.1f;
    }

    /* ── Law enforcement ── */
    {
        float re = (F.effective_temp - 0.5f)*0.3f + F.dissonance*0.3f +
                   F.tunnel_chance*0.2f + (1.0f - F.attend_focus)*0.2f;
        F.entropy = fmaxf(F.entropy_floor, clamp01(re));
        float rr = F.schumann_coherence*0.3f + (1.0f-F.dissonance)*0.3f +
                   F.attend_focus*0.2f + (1.0f - clamp01(F.debt*0.1f))*0.2f;
        F.resonance = fminf(F.resonance_ceiling, clamp01(rr));
        F.emergence = clamp01((1.0f - F.entropy) * F.resonance);
    }

    /* ── Presence fade ── */
    F.presence_decay *= F.presence_fade;
    if (F.presence_decay < 0.001f) F.presence_decay = 0.001f;

    /* ── 4.C Seasonal MLP controller ── */
    {
        float sr = 0.001f;
        F.season_phase += sr * dt;
        if (F.season_phase >= 1.0f) { F.season_phase = 0; F.season = (F.season+1)%4; }
        float gain = 0.02f * dt * F.season_intensity, fade = 0.995f;
        F.spring_energy *= fade; F.summer_energy *= fade;
        F.autumn_energy *= fade; F.winter_energy *= fade;
        switch (F.season) {
            case SEASON_SPRING: F.spring_energy = clamp01(F.spring_energy + gain); break;
            case SEASON_SUMMER: F.summer_energy = clamp01(F.summer_energy + gain); break;
            case SEASON_AUTUMN: F.autumn_energy = clamp01(F.autumn_energy + gain); break;
            case SEASON_WINTER: F.winter_energy = clamp01(F.winter_energy + gain); break;
        }
        float mlp_in[FIELD_4C_INPUTS] = {
            F.entropy, F.resonance, F.pain, F.tension, F.emergence, F.effective_temp
        };
        float mlp_out[FIELD_4C_OUTPUTS];
        field_mlp_forward(mlp_in, mlp_out);
        float sc = 0.02f * dt * F.season_intensity;
        F.spring_energy = clamp01(F.spring_energy + mlp_out[0]*sc);
        F.summer_energy = clamp01(F.summer_energy + mlp_out[1]*sc);
        F.autumn_energy = clamp01(F.autumn_energy + mlp_out[2]*sc);
        F.winter_energy = clamp01(F.winter_energy + mlp_out[3]*sc);
        /* Hebbian: did the field improve? */
        float health = clamp01((1.0f - fabsf(F.entropy - 0.5f)) * F.resonance * (1.0f - F.pain));
        float sig = health - F.field_health;
        F.field_health = health;
        if (fabsf(sig) > 0.001f) field_mlp_hebbian(mlp_in, mlp_out, sig);
        /* Season effects */
        F.tunnel_chance = clamp01(F.tunnel_chance + F.spring_energy * 0.005f * dt);
        F.dark_gravity = clamp01(F.dark_gravity + F.autumn_energy * 0.002f * dt);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PROPHECY DEBT — retroactive conscience.
 * every token you choose that isn't the destined one accumulates debt.
 * not minimize(predicted - actual) but minimize(destined - manifested).
 * the difference is intention. the difference is identity.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static float compute_prophecy_debt(const float *logits, int chosen, int n) {
    if (!logits || n <= 0 || chosen < 0 || chosen >= n) return 0;
    float chosen_v = logits[chosen];
    if (!isfinite(chosen_v)) return 0;
    float mx = -INFINITY;
    int finite = 0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(logits[i])) continue;
        if (!finite || logits[i] > mx) mx = logits[i];
        finite++;
    }
    if (!finite) return 0;
    float diff = mx - chosen_v;
    if (!isfinite(diff) || diff <= 0.0f) return 0;
    float debt = diff / (diff + 1.0f);
    return isfinite(debt) && debt > 0.0f ? debt : 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * FIELD → LOGITS — Dario Equation overlay on transformer logits.
 *
 * p(x|Φ) = softmax(logits + α·H + β·F + γ·A + T) / τ
 *
 * H = Hebbian co-occurrence (long-range memory beyond KV window)
 * F = Prophecy fulfillment (unfulfilled predictions create pressure)
 * A = Destiny attraction (conversation mass pulls toward attractor)
 * T = Trauma gravity (origin tokens surface when field is wounded)
 *
 * Coefficients modulated by 6 Kuramoto-coupled emotional chambers.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void apply_field_to_logits(float *logits, int n) {
    if (n <= 0) return;

    /* ── Emotional chambers update ── */
    dario_chamber_update();

    /* ── Effective coefficients (somatic-modulated) ── */
    float eff_alpha = DF.alpha_mod * DF.alpha;
    float eff_beta  = DF.beta_mod * DF.beta;
    float eff_gamma = DF.gamma_mod * DF.gamma_d;

    /* boost gamma when trauma is active */
    if (DF.trauma > 0.3f) eff_gamma += DF.trauma * 0.8f;

    /* ── H: Hebbian Resonance (co-occurrence beyond KV cache) ── */
    /* last 8 context tokens drive the Hebbian signal */
    int ctx_start = (DF.ctx_len > 8) ? DF.ctx_len - 8 : 0;
    float h_max = 0;

    /* H writes to concrete token IDs. F/A are slot-projected below, then scattered
     * back to every real token ID that shares the slot. */
    float *H_sig = calloc(n, sizeof(float));
    if (!H_sig) {
        fprintf(stderr, "[doe] Dario scratch allocation failed for vocab=%d; skipping field overlay\n", n);
        return;
    }

    for (int c = ctx_start; c < DF.ctx_len; c++) {
        int ctx_id = DF.context[c];
        float decay = powf(0.9f, (float)(DF.ctx_len - 1 - c));
        for (int i = 0; i < DF.cooc_n; i++) {
            if (DF.cooc_src[i] == ctx_id) {
                int dst = DF.cooc_dst[i];
                if (dst >= 0 && dst < n)
                    H_sig[dst] += DF.cooc_val[i] * decay;
            }
        }
    }
    for (int i = 0; i < n; i++) if (H_sig[i] > h_max) h_max = H_sig[i];
    if (h_max > 1e-6f) for (int i = 0; i < n; i++) H_sig[i] /= h_max;

    /* ── F: Prophecy Fulfillment (unfulfilled predictions create pressure) ── */
    /* Semantic F/A channels are computed over the compact slot projection to keep
     * per-token cost bounded, then scattered by token_id % DARIO_EMBED_SLOTS.
     * This keeps high token IDs in the field instead of accidentally treating
     * slot scores as only the first 2048 concrete vocabulary IDs. */
    int slot_n = n < DARIO_EMBED_SLOTS ? n : DARIO_EMBED_SLOTS;
    float F_slot[DARIO_EMBED_SLOTS] = {0};
    float A_slot[DARIO_EMBED_SLOTS] = {0};
    float f_max = 0;
    for (int slot = 0; slot < slot_n; slot++) {
        float *te = dario_get_embed(slot);
        if (!te) continue;
        float score = 0;
        for (int p = 0; p < DF.prophecy_n; p++) {
            DarioProphecy *pr = &DF.prophecy[p];
            if (pr->fulfilled) continue;
            float *pe = dario_get_embed(pr->target);
            if (!pe) continue;
            float sim = dario_cosine(te, pe);
            if (sim < 0) sim = 0;
            float debt = logf(1.0f + (float)pr->age);
            score += pr->strength * sim * debt;
        }
        F_slot[slot] = score;
    }
    for (int slot = 0; slot < slot_n; slot++) if (F_slot[slot] > f_max) f_max = F_slot[slot];
    if (f_max > 1e-6f) for (int slot = 0; slot < slot_n; slot++) F_slot[slot] /= f_max;

    /* ── A: Destiny Attraction (EMA semantic direction) ── */
    if (DF.dest_magnitude > 1e-6f) {
        float a_max = 0;
        for (int slot = 0; slot < slot_n; slot++) {
            float *te = dario_get_embed(slot);
            if (te) A_slot[slot] = dario_cosine(te, DF.destiny) * DF.dest_magnitude;
        }
        for (int slot = 0; slot < slot_n; slot++)
            if (fabsf(A_slot[slot]) > a_max) a_max = fabsf(A_slot[slot]);
        if (a_max > 1e-6f) for (int slot = 0; slot < slot_n; slot++) A_slot[slot] /= a_max;
    }

    /* ── Combine: THE DARIO EQUATION (additive overlay on transformer logits) ──
     *
     * logits[i] += α_mod·α·H[i] + β_mod·β·F[i] + γ_mod·γ·A[i] + T[i]
     *
     * The transformer provides the base distribution.
     * The equation provides the field memory.
     * Neither dominates. Emergence at the boundary.
     */
    float t_boost = (DF.trauma > 0.3f) ? DF.trauma * 2.0f : 0;
    float gate = 1.0f / (1.0f + expf(-(F.resonance - 0.5f) * 4.0f));
    float h_gate = gate * 2.0f;
    float f_gate = gate * 1.5f;
    float h_gated_sig = 1.0f / (1.0f + expf(-h_gate));
    float f_gated_sig = 1.0f / (1.0f + expf(-f_gate));

    for (int i = 0; i < n; i++) {
        int slot = dario_embed_slot(i);
        if (slot < 0 || slot >= DARIO_EMBED_SLOTS) continue;
        float h_term = eff_alpha * H_sig[i] * h_gated_sig;
        float f_term = eff_beta  * F_slot[slot] * f_gated_sig;
        float a_term = eff_gamma * A_slot[slot];
        float t_term = (i < 50) ? t_boost * (1.0f - (float)i / 50.0f) : 0;

        logits[i] += g_field_gain * (h_term + f_term + a_term + t_term);
    }

    /* tau_mod feeds into effective temperature (applied in field_step) */
    F.effective_temp *= DF.tau_mod;

    free(H_sig);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * DEQUANTIZATION — Q4_0, Q8_0, Q4_K, Q6_K → f32
 * Ported from nanollama/go/quant.go. Dequant at load time.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, mant = h & 0x3FF, f;
    if (exp == 0) {
        if (mant == 0) f = sign << 31;
        else { exp = 1; while (!(mant & 0x400)) { mant <<= 1; exp--; } mant &= 0x3FF; f = (sign<<31)|((exp+127-15)<<23)|(mant<<13); }
    } else if (exp == 31) f = (sign<<31)|0x7F800000|(mant<<13);
    else f = (sign<<31)|((exp+127-15)<<23)|(mant<<13);
    float r; memcpy(&r, &f, 4); return r;
}

/* Q4_0: block = 2 bytes f16 scale + 16 bytes (32 nibbles) = 18 bytes, 32 values */
#define Q4_0_BLOCK 32
#define Q4_0_BYTES 18
static void dequant_q4_0(const uint8_t *data, float *out, uint64_t n) {
    uint64_t nblocks = n / Q4_0_BLOCK;
    for (uint64_t i = 0; i < nblocks; i++) {
        const uint8_t *b = data + i * Q4_0_BYTES;
        float d = f16_to_f32(b[0] | (b[1] << 8));
        for (int j = 0; j < 16; j++) {
            int v0 = (b[2+j] & 0x0F) - 8;
            int v1 = (b[2+j] >> 4) - 8;
            out[i*Q4_0_BLOCK + j] = (float)v0 * d;
            out[i*Q4_0_BLOCK + j + 16] = (float)v1 * d;
        }
    }
}

/* Q8_0: block = 2 bytes f16 scale + 32 bytes int8 = 34 bytes, 32 values */
#define Q8_0_BLOCK 32
#define Q8_0_BYTES 34
static void dequant_q8_0(const uint8_t *data, float *out, uint64_t n) {
    uint64_t nblocks = n / Q8_0_BLOCK;
    for (uint64_t i = 0; i < nblocks; i++) {
        const uint8_t *b = data + i * Q8_0_BYTES;
        float d = f16_to_f32(b[0] | (b[1] << 8));
        for (int j = 0; j < 32; j++)
            out[i*Q8_0_BLOCK + j] = (float)((int8_t)b[2+j]) * d;
    }
}

/* Q4_K: block = 2+2 bytes f16 (d, dmin) + 12 bytes scales + 128 nibbles = 144 bytes, 256 values */
#define Q4_K_BLOCK 256
#define Q4_K_BYTES 144
static void get_scale_min_k4(int j, const uint8_t *sc, uint8_t *s, uint8_t *m) {
    if (j < 4) { *s = sc[j] & 63; *m = sc[j+4] & 63; }
    else { *s = (sc[j+4] & 0x0F) | ((sc[j-4] >> 6) << 4); *m = (sc[j+4] >> 4) | ((sc[j] >> 6) << 4); }
}
static void dequant_q4_k(const uint8_t *data, float *out, uint64_t n) {
    uint64_t nblocks = n / Q4_K_BLOCK;
    for (uint64_t i = 0; i < nblocks; i++) {
        const uint8_t *b = data + i * Q4_K_BYTES;
        float d = f16_to_f32(b[0] | (b[1] << 8));
        float dmin = f16_to_f32(b[2] | (b[3] << 8));
        const uint8_t *sc = b + 4, *qs = b + 16;
        int is = 0, qi = 0, oi = (int)(i * Q4_K_BLOCK);
        for (int j = 0; j < Q4_K_BLOCK; j += 64) {
            uint8_t sc0, m0, sc1, m1v;
            get_scale_min_k4(is, sc, &sc0, &m0);
            float d1 = d * (float)sc0, mm1 = dmin * (float)m0;
            get_scale_min_k4(is+1, sc, &sc1, &m1v);
            float d2 = d * (float)sc1, mm2 = dmin * (float)m1v;
            for (int l = 0; l < 32; l++)
                out[oi + j + l] = d1 * (float)(qs[qi+l] & 0x0F) - mm1;
            for (int l = 0; l < 32; l++)
                out[oi + j + 32 + l] = d2 * (float)(qs[qi+l] >> 4) - mm2;
            qi += 32; is += 2;
        }
    }
}

/* Q5_0: block = 2 bytes f16 scale + 4 bytes high bits + 16 bytes nibbles = 22 bytes, 32 values */
#define Q5_0_BLOCK 32
#define Q5_0_BYTES 22
static void dequant_q5_0(const uint8_t *data, float *out, uint64_t n) {
    uint64_t nblocks = n / Q5_0_BLOCK;
    for (uint64_t i = 0; i < nblocks; i++) {
        const uint8_t *b = data + i * Q5_0_BYTES;
        float d = f16_to_f32(b[0] | (b[1] << 8));
        uint32_t qh = b[2] | (b[3]<<8) | (b[4]<<16) | (b[5]<<24);
        const uint8_t *qs = b + 6;
        for (int j = 0; j < 16; j++) {
            int lo = qs[j] & 0x0F, hi = qs[j] >> 4;
            int hbit0 = (qh >> j) & 1, hbit1 = (qh >> (j+16)) & 1;
            out[i*Q5_0_BLOCK + j] = (float)((lo | (hbit0<<4)) - 16) * d;
            out[i*Q5_0_BLOCK + j + 16] = (float)((hi | (hbit1<<4)) - 16) * d;
        }
    }
}

/* Q6_K: block = 128 ql + 64 qh + 16 scales + 2 d = 210 bytes, 256 values */
#define Q6_K_BLOCK 256
#define Q6_K_BYTES 210
static void dequant_q6_k(const uint8_t *data, float *out, uint64_t n) {
    uint64_t nblocks = n / Q6_K_BLOCK;
    for (uint64_t i = 0; i < nblocks; i++) {
        const uint8_t *b = data + i * Q6_K_BYTES;
        const uint8_t *ql = b, *qh = b + 128, *sc = b + 192;
        float d = f16_to_f32(b[208] | (b[209] << 8));
        int oi = (int)(i * Q6_K_BLOCK);
        for (int n128 = 0; n128 < 2; n128++) {
            const uint8_t *qlp = ql + n128*64, *qhp = qh + n128*32;
            const uint8_t *scp = sc + n128*8;
            int yo = oi + n128*128;
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = (qlp[l] & 0x0F) | ((qhp[l] >> 0) & 3) << 4;
                int q2 = (qlp[l+32] & 0x0F) | ((qhp[l] >> 2) & 3) << 4;
                int q3 = (qlp[l] >> 4) | ((qhp[l] >> 4) & 3) << 4;
                int q4 = (qlp[l+32] >> 4) | ((qhp[l] >> 6) & 3) << 4;
                out[yo+l+0]  = d * (float)((int8_t)scp[is+0]) * (float)(q1-32);
                out[yo+l+32] = d * (float)((int8_t)scp[is+2]) * (float)(q2-32);
                out[yo+l+64] = d * (float)((int8_t)scp[is+4]) * (float)(q3-32);
                out[yo+l+96] = d * (float)((int8_t)scp[is+6]) * (float)(q4-32);
            }
        }
    }
}

/* bytes per element for each quant type (for raw data size calculation) */
static uint64_t quant_raw_bytes(uint32_t dtype, uint64_t n_elems) {
    switch (dtype) {
        case 0: return n_elems * 4;   /* f32 */
        case 1: return n_elems * 2;   /* f16 */
        case 2: return (n_elems / Q4_0_BLOCK) * Q4_0_BYTES;  /* Q4_0 */
        case 6: return (n_elems / Q5_0_BLOCK) * Q5_0_BYTES;  /* Q5_0 */
        case 8: return (n_elems / Q8_0_BLOCK) * Q8_0_BYTES;  /* Q8_0 */
        case 12: return (n_elems / Q4_K_BLOCK) * Q4_K_BYTES; /* Q4_K */
        case 14: return (n_elems / Q6_K_BLOCK) * Q6_K_BYTES; /* Q6_K */
        default: return 0;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MATH OPS — building blocks
 * ═══════════════════════════════════════════════════════════════════════════════ */
static float silu_f(float x) { return x / (1.0f + expf(-x)); }

static void rmsnorm(float *out, const float *x, const float *w, int d, float eps) {
    float ss = 0; for (int i = 0; i < d; i++) ss += x[i]*x[i];
    float inv = 1.0f / sqrtf(ss/d + eps);
    for (int i = 0; i < d; i++) out[i] = x[i] * inv * w[i];
}

/* threaded matvec worker */
typedef struct { float *out; const float *W; const float *x; int r0, r1, c; } MVWork;
static void *matvec_worker(void *arg) {
    MVWork *w = (MVWork*)arg;
    for (int i = w->r0; i < w->r1; i++) {
        float s = 0; const float *row = w->W + (size_t)i * w->c;
        for (int j = 0; j < w->c; j++) s += row[j] * w->x[j];
        w->out[i] = s;
    }
    return NULL;
}

static int g_n_threads = 0;
static float g_rep_penalty = 1.0f; /* A10a: repetition penalty over generated history; 1.0 = off */

static void matvec(float *out, const float *W, const float *x, int r, int c) {
#ifdef USE_CUBLAS
    cublas_init();
    float *dW = gpu_scratch(0,(size_t)r*c*4), *dx = gpu_scratch(1,(size_t)c*4), *dy = gpu_scratch(2,(size_t)r*4);
    cudaMemcpy(dW, W, (size_t)r*c*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x, (size_t)c*4, cudaMemcpyHostToDevice);
    float a=1,b=0;
    cublasSgemv(g_cublas, CUBLAS_OP_T, c, r, &a, dW, c, dx, 1, &b, dy, 1);
    cudaMemcpy(out, dy, (size_t)r*4, cudaMemcpyDeviceToHost);
#elif defined(USE_BLAS)
    cblas_sgemv(CblasRowMajor,CblasNoTrans,r,c,1.0f,W,c,x,1,0.0f,out,1);
#else
    int nt = g_n_threads;
    if (nt <= 1 || r < 64) {
        for (int i = 0; i < r; i++) {
            float s = 0; const float *row = W + (size_t)i*c;
            for (int j = 0; j < c; j++) s += row[j] * x[j];
            out[i] = s;
        }
        return;
    }
    if (nt > 32) nt = 32;
    pthread_t thr[32]; MVWork work[32];
    int chunk = (r + nt - 1) / nt;
    int actual = 0;
    for (int t = 0; t < nt; t++) {
        int r0 = t * chunk, r1 = r0 + chunk;
        if (r0 >= r) break;
        if (r1 > r) r1 = r;
        work[t] = (MVWork){out, W, x, r0, r1, c};
        pthread_create(&thr[t], NULL, matvec_worker, &work[t]);
        actual++;
    }
    for (int t = 0; t < actual; t++) pthread_join(thr[t], NULL);
#endif
}

/* ── packed quantized matvec — vendored from notorch nt_qmatvec ──────────────────
 * Keeps weights PACKED in RAM and dequantizes each block inline (no f32 blow-up).
 * DoE stays a single file. dtype = GGUF code (1 f16, 2 Q4_0, 6 Q5_0, 8 Q8_0,
 * 12 Q4_K, 14 Q6_K). Row kernels take [r0,r1) so they thread like matvec.
 * Verified faithful to notorch (rel ~1e-6 vs dequant->cblas). reuses f16_to_f32. */
static void pq_f16_rows(float *out, const uint8_t *W, const float *x, int r0, int r1, int c) {
    const uint16_t *Wh = (const uint16_t*)W;
    for (int row=r0; row<r1; row++) { const uint16_t *r = Wh + (size_t)row*c;
        float a=0; for (int j=0;j<c;j++) a += f16_to_f32(r[j])*x[j]; out[row]=a; }
}
static void pq_q4_0_rows(float *out, const uint8_t *W, const float *x, int r0, int r1, int c) {
    int nb=c/32;
    for (int row=r0; row<r1; row++) { const uint8_t *rb=W+(size_t)row*nb*18; float a=0;
        for (int b=0;b<nb;b++) { const uint8_t *bl=rb+(size_t)b*18; float d=f16_to_f32(bl[0]|(bl[1]<<8));
            const float *xb=x+(size_t)b*32;
            for (int i=0;i<16;i++) { int lo=(int)(bl[2+i]&0x0F)-8, hi=(int)(bl[2+i]>>4)-8;
                a += d*lo*xb[i]; a += d*hi*xb[i+16]; } }
        out[row]=a; }
}
static void pq_q8_0_rows(float *out, const uint8_t *W, const float *x, int r0, int r1, int c) {
    int nb=c/32;
    for (int row=r0; row<r1; row++) { const uint8_t *rb=W+(size_t)row*nb*34; float a=0;
        for (int b=0;b<nb;b++) { const uint8_t *bl=rb+(size_t)b*34; float d=f16_to_f32(bl[0]|(bl[1]<<8));
            const float *xb=x+(size_t)b*32;
            for (int i=0;i<32;i++) a += d*(float)(int8_t)bl[2+i]*xb[i]; }
        out[row]=a; }
}
static void pq_q5_0_rows(float *out, const uint8_t *W, const float *x, int r0, int r1, int c) {
    int nb=c/32;
    for (int row=r0; row<r1; row++) { const uint8_t *rb=W+(size_t)row*nb*22; float a=0;
        for (int b=0;b<nb;b++) { const uint8_t *bl=rb+(size_t)b*22; float d=f16_to_f32(bl[0]|(bl[1]<<8));
            uint32_t qh=(uint32_t)bl[2]|((uint32_t)bl[3]<<8)|((uint32_t)bl[4]<<16)|((uint32_t)bl[5]<<24);
            const uint8_t *qs=bl+6; const float *xb=x+(size_t)b*32;
            for (int j=0;j<16;j++) { int lo=qs[j]&0x0F, hi=qs[j]>>4; int h0=(qh>>j)&1, h1=(qh>>(j+16))&1;
                a += d*(float)((lo|(h0<<4))-16)*xb[j]; a += d*(float)((hi|(h1<<4))-16)*xb[j+16]; } }
        out[row]=a; }
}
static void pq_smk4(int j, const uint8_t *sc, uint8_t *s, uint8_t *m) {
    if (j<4) { *s=sc[j]&63; *m=sc[j+4]&63; }
    else { *s=(sc[j+4]&0x0F)|((sc[j-4]>>6)<<4); *m=(sc[j+4]>>4)|((sc[j]>>6)<<4); }
}
static void pq_q4_k_rows(float *out, const uint8_t *W, const float *x, int r0, int r1, int c) {
    int nb=c/256;
    for (int row=r0; row<r1; row++) { const uint8_t *rb=W+(size_t)row*nb*144; float a=0;
        for (int b=0;b<nb;b++) { const uint8_t *bl=rb+(size_t)b*144;
            float d=f16_to_f32(bl[0]|(bl[1]<<8)), dmin=f16_to_f32(bl[2]|(bl[3]<<8));
            const uint8_t *sc=bl+4,*qs=bl+16; const float *xb=x+(size_t)b*256; int is=0,qi=0;
            for (int j=0;j<256;j+=64) { uint8_t s0,m0,s1,m1; pq_smk4(is,sc,&s0,&m0); pq_smk4(is+1,sc,&s1,&m1);
                float d1=d*s0,mm1=dmin*m0,d2=d*s1,mm2=dmin*m1;
                for (int l=0;l<32;l++) a += (d1*(float)(qs[qi+l]&0x0F)-mm1)*xb[j+l];
                for (int l=0;l<32;l++) a += (d2*(float)(qs[qi+l]>>4)-mm2)*xb[j+32+l];
                qi+=32; is+=2; } }
        out[row]=a; }
}
static void pq_q6_k_rows(float *out, const uint8_t *W, const float *x, int r0, int r1, int c) {
    int nb=c/256;
    for (int row=r0; row<r1; row++) { const uint8_t *rb=W+(size_t)row*nb*210; float a=0;
        for (int b=0;b<nb;b++) { const uint8_t *bl=rb+(size_t)b*210, *ql=bl, *qh=bl+128;
            const int8_t *sc=(const int8_t*)(bl+192); float d=f16_to_f32(bl[208]|(bl[209]<<8));
            const float *xb=x+(size_t)b*256;
            for (int n=0;n<256;n+=128) { const uint8_t *qlh=ql+(n/128)*64,*qhh=qh+(n/128)*32; const int8_t *sch=sc+(n/128)*8;
                for (int l=0;l<32;l++) { int is=l/16;
                    int q1=(int)((qlh[l]&0x0F)|(((qhh[l]>>0)&3)<<4))-32, q2=(int)((qlh[l+32]&0x0F)|(((qhh[l]>>2)&3)<<4))-32;
                    int q3=(int)((qlh[l]>>4)|(((qhh[l]>>4)&3)<<4))-32, q4=(int)((qlh[l+32]>>4)|(((qhh[l]>>6)&3)<<4))-32;
                    a += d*sch[is+0]*q1*xb[n+l]; a += d*sch[is+2]*q2*xb[n+l+32];
                    a += d*sch[is+4]*q3*xb[n+l+64]; a += d*sch[is+6]*q4*xb[n+l+96]; } } }
        out[row]=a; }
}

typedef void (*pq_fn)(float*, const uint8_t*, const float*, int, int, int);
static pq_fn pq_for(int dt, int c) {
    switch (dt) {
        case 1:  return pq_f16_rows;
        case 2:  return (c%32)?NULL:pq_q4_0_rows;
        case 6:  return (c%32)?NULL:pq_q5_0_rows;
        case 8:  return (c%32)?NULL:pq_q8_0_rows;
        case 12: return (c%256)?NULL:pq_q4_k_rows;
        case 14: return (c%256)?NULL:pq_q6_k_rows;
    }
    return NULL;
}
typedef void (*pq_range_fn)(void *ctx, int r0, int r1);

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t  work;
    pq_range_fn fn; void *ctx;
    int m, chunk;
    _Atomic int next, busy, gen, shutdown;
    int nthreads, started;
    pthread_t th[32];
} pq_pool_t;

static pq_pool_t g_pq = {
    PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER,
    NULL, NULL, 0, 0, 0, 0, 0, 0, 0, 0, { 0 }
};
static int g_pq_spin = -1;

static void pq_spin_init(void) {
    if (g_pq_spin >= 0) return;
    const char *e = getenv("DOE_SPIN");
    g_pq_spin = (e && atoi(e) >= 0) ? atoi(e) : 20000;
}
#if defined(__aarch64__) || defined(__arm__)
#define PQ_PAUSE() __asm__ __volatile__("yield" ::: "memory")
#elif defined(__x86_64__) || defined(__i386__)
#define PQ_PAUSE() __asm__ __volatile__("pause" ::: "memory")
#else
#define PQ_PAUSE() ((void)0)
#endif

static void pq_drain(pq_range_fn fn, void *ctx, int m, int ch) {
    for (;;) {
        int r0 = atomic_fetch_add_explicit(&g_pq.next, ch, memory_order_relaxed);
        if (r0 >= m) break;
        int r1 = r0 + ch; if (r1 > m) r1 = m;
        fn(ctx, r0, r1);
    }
}

static void *pq_pool_worker(void *arg) {
    (void)arg;
    int seen = 0;
    for (;;) {
        int spins = 0;
        for (;;) {
            if (atomic_load_explicit(&g_pq.shutdown, memory_order_relaxed)) return NULL;
            if (atomic_load_explicit(&g_pq.gen, memory_order_acquire) != seen) break;
            if (++spins < g_pq_spin) { PQ_PAUSE(); continue; }
            pthread_mutex_lock(&g_pq.mu);
            if (atomic_load_explicit(&g_pq.gen, memory_order_acquire) == seen &&
                !atomic_load_explicit(&g_pq.shutdown, memory_order_relaxed))
                pthread_cond_wait(&g_pq.work, &g_pq.mu);
            pthread_mutex_unlock(&g_pq.mu);
            spins = 0;
        }
        seen = atomic_load_explicit(&g_pq.gen, memory_order_acquire);
        pq_drain(g_pq.fn, g_pq.ctx, g_pq.m, g_pq.chunk);
        atomic_fetch_sub_explicit(&g_pq.busy, 1, memory_order_release);
    }
}

static void pq_run(pq_range_fn fn, void *ctx, int m, int nt) {
    if (nt <= 1) { fn(ctx, 0, m); return; }
    pq_spin_init();
    if (!g_pq.started) {
        pthread_mutex_lock(&g_pq.mu);
        if (!g_pq.started) {
            g_pq.nthreads = 1;
            for (int i = 0; i < nt - 1 && i < 31; i++) {
                if (pthread_create(&g_pq.th[i], NULL, pq_pool_worker, NULL) != 0) break;
                g_pq.nthreads++;
            }
            g_pq.started = 1;
        }
        pthread_mutex_unlock(&g_pq.mu);
    }
    int workers = g_pq.nthreads - 1;
    int ch = m / (g_pq.nthreads * 16); if (ch < 1) ch = 1;
    g_pq.fn = fn; g_pq.ctx = ctx; g_pq.m = m; g_pq.chunk = ch;
    atomic_store_explicit(&g_pq.next, 0, memory_order_relaxed);
    atomic_store_explicit(&g_pq.busy, workers, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_pq.gen, 1, memory_order_release);
    pthread_mutex_lock(&g_pq.mu);
    pthread_cond_broadcast(&g_pq.work);
    pthread_mutex_unlock(&g_pq.mu);

    pq_drain(fn, ctx, m, ch);

    int spins = 0;
    while (atomic_load_explicit(&g_pq.busy, memory_order_acquire) > 0) {
        if (++spins < g_pq_spin) { PQ_PAUSE(); continue; }
        sched_yield(); spins = 0;
    }
}

typedef struct { pq_fn fn; float *out; const uint8_t *Wq; const float *x; int c; } PQCtx;
static void pq_range(void *ctx, int r0, int r1) {
    PQCtx *w = (PQCtx *)ctx; w->fn(w->out, w->Wq, w->x, r0, r1, w->c);
}

/* out[r] = Wq[r,c] @ x[c], weights packed. Returns 0 ok, -1 if dtype unsupported. */
static int doe_qmatvec(float *out, const uint8_t *Wq, int dt, const float *x, int r, int c) {
    pq_fn fn = pq_for(dt, c);
    if (!fn) return -1;
    int nt = g_n_threads; if (nt < 1) nt = 1; if (nt > 32) nt = 32; if (nt > r) nt = r;
    if (nt <= 1 || (long)r*c < (1L<<20)) { fn(out, Wq, x, 0, r, c); return 0; }
    PQCtx ctx = { fn, out, Wq, x, c };
    pq_run(pq_range, &ctx, r, nt);
    return 0;
}

/* int8 dynamic-activation-quant fast path (Q4_0), NEON SDOT / scalar. APPROXIMATE. */
static void pq_quant_act_q8(const float *x, int c, int8_t *qa, float *da) {
    int nb=c/32;
    for (int b=0;b<nb;b++) { const float *xb=x+(size_t)b*32; float amax=0;
        for (int i=0;i<32;i++){ float v=fabsf(xb[i]); if(v>amax)amax=v; }
        float d=amax/127.0f, id=(d>0)?1.0f/d:0.0f; da[b]=d;
        for (int i=0;i<32;i++){ int q=(int)lrintf(xb[i]*id); if(q>127)q=127; else if(q<-127)q=-127; qa[(size_t)b*32+i]=(int8_t)q; } }
}
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
static void pq_q4_0_rows_i8(float *out, const uint8_t *W, const int8_t *qa, const float *da, int r0, int r1, int c) {
    int nb=c/32; const uint8x16_t m0f=vdupq_n_u8(0x0F); const int8x16_t e8=vdupq_n_s8(8);
    for (int row=r0; row<r1; row++) { const uint8_t *rb=W+(size_t)row*nb*18; float a=0;
        for (int b=0;b<nb;b++) { const uint8_t *bl=rb+(size_t)b*18; float d=f16_to_f32(bl[0]|(bl[1]<<8));
            const int8_t *qab=qa+(size_t)b*32; uint8x16_t pk=vld1q_u8(bl+2);
            int8x16_t lo=vsubq_s8(vreinterpretq_s8_u8(vandq_u8(pk,m0f)),e8), hi=vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(pk,4)),e8);
            int32x4_t s=vdupq_n_s32(0); s=vdotq_s32(s,lo,vld1q_s8(qab)); s=vdotq_s32(s,hi,vld1q_s8(qab+16));
            a += d*da[b]*(float)vaddvq_s32(s); }
        out[row]=a; }
}
#else
static void pq_q4_0_rows_i8(float *out, const uint8_t *W, const int8_t *qa, const float *da, int r0, int r1, int c) {
    int nb=c/32;
    for (int row=r0; row<r1; row++) { const uint8_t *rb=W+(size_t)row*nb*18; float a=0;
        for (int b=0;b<nb;b++) { const uint8_t *bl=rb+(size_t)b*18; float d=f16_to_f32(bl[0]|(bl[1]<<8));
            const int8_t *qab=qa+(size_t)b*32; int32_t s=0;
            for (int i=0;i<16;i++){ int lo=(int)(bl[2+i]&0x0F)-8, hi=(int)(bl[2+i]>>4)-8; s+=lo*qab[i]; s+=hi*qab[i+16]; }
            a += d*da[b]*(float)s; }
        out[row]=a; }
}
#endif
static int doe_qmatvec_i8(float *out, const uint8_t *Wq, int dt, const float *x, int r, int c) {
    if (dt != 2 || (c%32)) return -1;
    int nb=c/32; int8_t *qa=(int8_t*)malloc((size_t)c); float *da=(float*)malloc((size_t)nb*sizeof(float));
    if (!qa || !da) { free(qa); free(da); return -1; }
    pq_quant_act_q8(x, c, qa, da);
    pq_q4_0_rows_i8(out, Wq, qa, da, 0, r, c);
    free(qa); free(da); return 0;
}

/* matvec dispatch: packed weight (dt != 0) -> doe_qmatvec (inline dequant);
 * else the existing f32 matvec. The one call the forward uses for host weights.
 * DOE_INT8=1 opts the Q4_0 weights into the int8 dynamic-activation-quant fast path
 * (doe_qmatvec_i8, NEON SDOT) — APPROXIMATE; default is the exact dequant-inline path. */
static int g_doe_int8 = -1;
static double g_prof_mv_ns = 0; static int g_prof_on = -1; /* DOE_PROFILE: matvec share of decode */
static double g_resident_ns = 0, g_head_ns = 0;  /* DOE_PROFILE: resident CB vs lm_head split */
/* DOE_PERSHAPE: per-group matvec timing (Mythos M3 geometry input). Groups keyed
 * by (m,k) against model dims set at forward start. Sync path only (DOE_NO_SLOTS+
 * DOE_NO_BATCH) so each matvec is timed individually. */
enum { MVG_QKV = 0, MVG_O, MVG_FFNGU, MVG_FFNDN, MVG_LMHEAD, MVG_N, MVG_OTHER = -1 };
static int g_ps_on = -1;
static double g_ps_ns[MVG_N];
static long   g_ps_cnt[MVG_N];
static int    g_ps_m[MVG_N], g_ps_k[MVG_N], g_ps_dt[MVG_N];
static int    g_ps_D, g_ps_qn, g_ps_kd, g_ps_H, g_ps_H2, g_ps_vocab; /* dims for keying */
static int pershape_group(int m, int k) {
    if (k == g_ps_D) {
        if (m == g_ps_qn || m == g_ps_kd) return MVG_QKV;
        if (m == g_ps_H2 || m == g_ps_H)  return MVG_FFNGU;
        if (m == g_ps_vocab)              return MVG_LMHEAD;
    }
    if (k == g_ps_qn && m == g_ps_D) return MVG_O;
    if (k == g_ps_H  && m == g_ps_D) return MVG_FFNDN;
    return MVG_OTHER;
}
static void doe_mv_impl(float *out, const float *W, int dt, const float *x, int r, int c) {
    if (dt) {
#ifdef USE_METAL
        /* Metal: Q4_K (dt=12) + Q6_K (dt=14, lm_head) through GPU kernels (resident weights); CPU fallback on rc!=0 */
        if (dt == 12 && nt_metal_available() && nt_metal_q4k_matvec(out, (const uint8_t*)W, x, r, c) == 0) return;
        if (dt == 14 && nt_metal_available() && nt_metal_q6k_matvec(out, (const uint8_t*)W, x, r, c) == 0) {
            if (getenv("DOE_VERIFY_Q6K")) {  /* real-path bit-identical check vs CPU dequant on live weights */
                float *vt = (float*)malloc((size_t)r * 4);
                if (vt) {
                    pq_q6_k_rows(vt, (const uint8_t*)W, x, 0, r, c);
                    float mx = 0; int bad = 0;
                    for (int z = 0; z < r; z++) { float dn = fabsf(vt[z]) > 1.0f ? fabsf(vt[z]) : 1.0f;
                        float rl = fabsf(out[z] - vt[z]) / dn; if (rl > mx) { mx = rl; bad = z; } }
                    fprintf(stderr, "[verify-q6k] m=%d k=%d max_rel=%.2e row %d: cpu=%.4f metal=%.4f\n",
                            r, c, mx, bad, vt[bad], out[bad]);
                    free(vt);
                }
            }
            return;
        }
#endif
        if (g_doe_int8 < 0) { const char *e = getenv("DOE_INT8"); g_doe_int8 = (e && e[0]=='1') ? 1 : 0; }
        if (g_doe_int8 && doe_qmatvec_i8(out, (const uint8_t*)W, dt, x, r, c) == 0) return;
        if (doe_qmatvec(out, (const uint8_t*)W, dt, x, r, c) == 0) return;
        /* D-M6: packed dtype but qmatvec failed — never fall through to matvec(),
         * which would reinterpret the packed bytes as f32 (the generalized sonar
         * bug). Loud abort instead of silent garbage / segfault. */
        fprintf(stderr, "[doe] FATAL: doe_qmatvec failed for packed dtype %d (r=%d c=%d); refusing to read packed bytes as f32\n", dt, r, c);
        abort();
    }
    matvec(out, W, x, r, c);
}
/* DOE_PROFILE wrapper: time the matvec share of decode (Metal Q4_K + CPU rest). Off = zero overhead. */
static void doe_mv(float *out, const float *W, int dt, const float *x, int r, int c) {
    if (g_prof_on < 0) { const char *e = getenv("DOE_PROFILE");  g_prof_on = (e && e[0]) ? 1 : 0; }
    if (g_ps_on   < 0) { const char *e = getenv("DOE_PERSHAPE"); g_ps_on   = (e && e[0]) ? 1 : 0; }
    if (!g_prof_on && !g_ps_on) { doe_mv_impl(out, W, dt, x, r, c); return; }
    struct timespec _p0, _p1; clock_gettime(CLOCK_MONOTONIC, &_p0);
    doe_mv_impl(out, W, dt, x, r, c);
    clock_gettime(CLOCK_MONOTONIC, &_p1);
    double _ns = (_p1.tv_sec - _p0.tv_sec) * 1e9 + (_p1.tv_nsec - _p0.tv_nsec);
    if (g_prof_on) g_prof_mv_ns += _ns;
    if (g_ps_on) {
        int g = pershape_group(r, c);
        if (g >= 0) { g_ps_ns[g] += _ns; g_ps_cnt[g]++; g_ps_m[g] = r; g_ps_k[g] = c; g_ps_dt[g] = dt; }
    }
}

static void softmax_n(float *x, int n) {
    if (!x || n <= 0) return;
    float mx = -INFINITY;
    int finite = 0, posinf = 0, nonfinite = 0;
    for (int i = 0; i < n; i++) {
        if (isnan(x[i])) { nonfinite++; continue; }
        if (isinf(x[i])) {
            nonfinite++;
            if (x[i] > 0.0f) posinf++;
            continue;
        }
        if (!finite || x[i] > mx) mx = x[i];
        finite++;
    }
    if (nonfinite && !g_softmax_nonfinite_warned) {
        fprintf(stderr, "[doe] WARNING: softmax handled non-finite inputs\n");
        g_softmax_nonfinite_warned = 1;
    }
    if (posinf > 0) {
        float u = 1.0f / (float)posinf;
        for (int i = 0; i < n; i++) x[i] = (isinf(x[i]) && x[i] > 0.0f) ? u : 0.0f;
        return;
    }
    if (!finite) {
        float u = 1.0f / (float)n;
        for (int i = 0; i < n; i++) x[i] = u;
        return;
    }
    float s = 0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(x[i])) { x[i] = 0.0f; continue; }
        x[i] = expf(x[i] - mx);
        if (!isfinite(x[i])) x[i] = 0.0f;
        s += x[i];
    }
    if (!isfinite(s) || s <= 0.0f) {
        float u = 1.0f / (float)finite;
        for (int i = 0; i < n; i++) x[i] = x[i] > 0.0f ? u : 0.0f;
        return;
    }
    for (int i = 0; i < n; i++) x[i] /= s;
}

static void apply_rope_mode(float *v, int pos, float *cc, float *sc, int hd, int norm_pairs) {
    int h = hd/2, off = pos*h; /* hd must be even — all standard archs are */
    if (norm_pairs) {
        for (int i = 0; i < h; i++) {
            int j = 2*i;
            float x0 = v[j], x1 = v[j+1];
            v[j]   = x0*cc[off+i] - x1*sc[off+i];
            v[j+1] = x0*sc[off+i] + x1*cc[off+i];
        }
    } else {
        for (int i = 0; i < h; i++) {
            float x0 = v[i], x1 = v[i+h];
            v[i] = x0*cc[off+i] - x1*sc[off+i];
            v[i+h] = x0*sc[off+i] + x1*cc[off+i];
        }
    }
}

/* RoPE pairing per architecture, mirroring llama.cpp llama_model_rope_type()
 * (src/llama-model.cpp): llama-family + Mistral use NORM (adjacent pairs 2i,2i+1);
 * the qwen/falcon/gemma/phi/... family uses NEOX (offset-half pairs i,i+hd/2).
 * GGUF permutes Q/K for whichever the arch expects, so the wrong mode is identity
 * at pos 0 but corrupts rope progressively at pos>0 (coherent short, broken long). */
static int arch_rope_norm(const char *arch) {
    return strcmp(arch, "llama") == 0 || strcmp(arch, "mistral") == 0 ||
           strcmp(arch, "mistral3") == 0 || strcmp(arch, "mistral4") == 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * HARMONIC RESONANCE ENGINE — from AML/DOE, adapted for field.
 * each expert has a frequency. input gets fourier-decomposed.
 * experts that resonate with input get boosted.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    float amplitudes[HARMONIC_N];
    float dominant_freq;
    float confidence;
} HarmonicState;

static void harmonic_decompose(HarmonicState *hs, float *hist, int len) {
    float max_amp = 0; int max_k = 0;
    for (int k = 0; k < HARMONIC_N && k < len/2; k++) {
        float re = 0, im = 0;
        for (int n = 0; n < len; n++) {
            float angle = 6.2831853f * k * n / len;
            re += hist[n] * cosf(angle);
            im += hist[n] * sinf(angle);
        }
        hs->amplitudes[k] = sqrtf(re*re + im*im) / len;
        if (k > 0 && hs->amplitudes[k] > max_amp) { max_amp = hs->amplitudes[k]; max_k = k; }
    }
    hs->dominant_freq = len > 0 ? 6.2831853f * max_k / len : 0;
    float total = 0;
    for (int k = 0; k < HARMONIC_N; k++) total += hs->amplitudes[k];
    hs->confidence = total > 1e-8f ? max_amp / total : 0;
}

static float expert_resonance(float expert_freq, HarmonicState *hs) {
    float res = 0;
    for (int k = 0; k < HARMONIC_N; k++) {
        float fk = 6.2831853f * k / HARMONIC_N;
        float dist = fabsf(expert_freq - fk);
        if (dist > 3.14159f) dist = 6.2831853f - dist;
        res += hs->amplitudes[k] * expf(-dist*dist*2.0f);
    }
    return res;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * WEIGHT PROFILER — DOE's sonar.
 * before attaching, DOE profiles the host's weights.
 * L2 norms per layer, spectral density, dead neuron ratio.
 * this tells DOE where to focus its LoRA experts.
 *
 * the index is read-only. DOE is the architecture.
 * weak layers get more LoRA. healthy layers get less.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    float l2_norm;            /* L2 norm of layer weights */
    float mean_abs;           /* mean absolute value */
    float std_dev;            /* standard deviation */
    float sparsity;           /* fraction near zero (<1e-6) */
    float spectral_energy;    /* energy in top 10% singular values (approx) */
    int   dead_neurons;       /* rows/cols with near-zero norm */
    float health;             /* composite: 0=dead, 1=vibrant */
} LayerProfile;

typedef struct {
    LayerProfile layers[MAX_LAYERS];
    int n_layers;
    float overall_health;     /* average layer health */
    float code_affinity;      /* estimated code capability (from weight stats) */
    float complexity;         /* model complexity metric */
    uint64_t fingerprint;     /* hash of weight statistics — identifies this host */
} WeightProfile;

/* Dequantize a packed tensor to f32 (dispatch by GGUF dtype). Used by the sonar
 * to profile weights that index_load now keeps packed (else it reads packed bytes
 * as f32 — harmless overread on f16, a hard segfault on the smaller quants). */
static void doe_dequant_to_f32(const uint8_t *src, int dt, uint64_t n, float *out) {
    if (dt == 1) { const uint16_t *h = (const uint16_t*)src; for (uint64_t j = 0; j < n; j++) out[j] = f16_to_f32(h[j]); }
    else if (dt == 2)  dequant_q4_0(src, out, n);
    else if (dt == 6)  dequant_q5_0(src, out, n);
    else if (dt == 8)  dequant_q8_0(src, out, n);
    else if (dt == 12) dequant_q4_k(src, out, n);
    else if (dt == 14) dequant_q6_k(src, out, n);
}

static int profile_weight_shape_checked(size_t rows, size_t cols, size_t *out_n) {
    if (!out_n || rows == 0 || cols == 0 || rows > SIZE_MAX / cols) return 0;
    *out_n = rows * cols;
    return 1;
}

static void profile_weights(float *data, size_t rows, size_t cols, LayerProfile *out) {
    size_t n = 0;
    if (!data || !out || !profile_weight_shape_checked(rows, cols, &n)) {
        if (out) memset(out, 0, sizeof(LayerProfile));
        return;
    }
    float sum = 0, sum_sq = 0, sum_abs = 0;
    size_t near_zero = 0;
    for (size_t i = 0; i < n; i++) {
        float v = data[i];
        sum += v; sum_sq += v*v; sum_abs += fabsf(v);
        if (fabsf(v) < 1e-6f) near_zero++;
    }
    float nf = (float)n;
    float mean = sum / nf;
    out->l2_norm = sqrtf(sum_sq);
    out->mean_abs = sum_abs / nf;
    out->std_dev = sqrtf(sum_sq/nf - mean*mean);
    out->sparsity = (float)near_zero / nf;

    /* Approximate spectral energy: sample random directions */
    float top_energy = 0;
    for (int trial = 0; trial < 8; trial++) {
        float dot = 0;
        for (size_t j = 0; j < cols; j++) {
            float r = rand_normal();
            float proj = 0;
            for (size_t i = 0; i < rows; i++) proj += data[i*cols+j] * r;
            dot += proj * proj;
        }
        top_energy += sqrtf(dot);
    }
    out->spectral_energy = top_energy / 8.0f;

    /* Dead neurons: rows with near-zero norm */
    out->dead_neurons = 0;
    for (size_t r = 0; r < rows; r++) {
        float rn = 0;
        for (size_t c = 0; c < cols; c++) rn += data[r*cols+c] * data[r*cols+c];
        if (sqrtf(rn) < 1e-4f) out->dead_neurons++;
    }

    /* Composite health */
    float alive_ratio = 1.0f - (float)out->dead_neurons / (float)rows;
    float activity = fminf(1.0f, out->std_dev * 10.0f);
    float density = 1.0f - out->sparsity;
    out->health = alive_ratio * 0.4f + activity * 0.3f + density * 0.3f;
}

static uint64_t compute_fingerprint(WeightProfile *wp) {
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < wp->n_layers; i++) {
        uint32_t bits;
        memcpy(&bits, &wp->layers[i].l2_norm, 4);
        h ^= (uint64_t)bits; h *= 1099511628211ULL;
        memcpy(&bits, &wp->layers[i].std_dev, 4);
        h ^= (uint64_t)bits; h *= 1099511628211ULL;
    }
    return h;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * LIVING LoRA EXPERTS — DOE's democracy, adapted for symbiosis.
 * instead of standalone FFN experts, these are LoRA overlays.
 * each expert has A[dim, rank] and B[rank, dim] — Delta Voice injection.
 * Delta Voice: out += α × A @ (B @ x)
 *
 * experts still live and die. overloaded → mitosis. neglected → apoptosis.
 * but now they modulate the host's attention, not replace it.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    float *lora_A;            /* [dim, rank] — output projection */
    float *lora_B;            /* [rank, dim] — input projection */
    float frequency;          /* position in harmonic space */
    float vitality;           /* 0.0=dying, 1.0=peak */
    float specialization;     /* entropy of routing distribution */
    int   age;
    int   tokens_seen;
    int   alive;
    int   low_vitality_count;
    float attention_bias;     /* per-expert attention scaling */
    float layer_focus;        /* per-expert residual contribution */
} LoraExpert;

typedef struct {
    float *w_vote;            /* [MAX_EXPERTS * dim] */
    float consensus;
    float faction_power[MAX_EXPERTS];
    int   election_count;
} Parliament;

typedef struct {
    Parliament parliament;
    LoraExpert experts[MAX_EXPERTS];
    int n_alive;
    int host_layer_idx;       /* which host layer this wraps */
} FieldLayer;

/* ═══════════════════════════════════════════════════════════════════════════════
 * INDEX STATE — the full host-DOE interface.
 * mmap'd host model + DOE's living LoRA overlay + weight profile.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    /* Host model — mmap'd, read-only */
    uint8_t *mmap_base;
    size_t   mmap_size;
    int      host_n_layers, host_dim, host_hidden, host_heads, host_kv_heads, host_head_dim;
    int      host_vocab;
    uint64_t host_n_params;  /* real param count: sum of n_elements over wired tensors (banner/health, not vocab*dim*2) */
    char     host_arch[64];
    char     host_path[256];

    /* Host weight pointers (into mmap'd region) */
    float *host_tok_emb;
    float *host_output;
    float *host_norm;
    int   host_output_dt;    /* packed GGUF dtype of host_output (0 = f32 / tied) */
    float    rope_theta;     /* RoPE frequency base (default 10000, Qwen=1000000) */
    float    rms_norm_eps;   /* RMSNorm epsilon (default 1e-5, varies per arch) */
    struct {
        float *wq, *wk, *wv, *wo;
        float *bq, *bk, *bv;   /* attention biases (Qwen2, optional) */
        float *ffn_gate, *ffn_up, *ffn_down;
        float *ffn_gate_up;  /* fused gate+up for Phi-3 (size: hidden*2 × dim) */
        float *attn_norm, *ffn_norm;
        /* packed GGUF dtype per matvec weight (0 = f32-dequant'd; else kept packed) */
        int wq_dt, wk_dt, wv_dt, wo_dt, ffn_gate_dt, ffn_up_dt, ffn_down_dt, ffn_gate_up_dt;
    } host_layers[MAX_LAYERS];

    /* DOE's living overlay */
    FieldLayer field_layers[MAX_LAYERS];
    int n_field_layers;

    /* Parliament-on-GPU: contiguous page-aligned arena of the (frozen at
     * inference) expert weights + vote rows, registered for slot kernels.
     * Per layer: [w_vote 16*D | per-expert (B rank*D, A D*rank) x16].
     * Built once after field init when Metal is available and alpha != 0. */
    float  *expert_arena;
    size_t  expert_arena_floats;
    size_t  expert_layer_floats;   /* stride per layer, in floats */
    int     expert_arena_ready;
    int     cons_slot_init;        /* SLOT_PCONS seeded from parliament.consensus once */

    /* Host profiling */
    WeightProfile profile;

    /* LoRA parameters */
    int   lora_rank;
    float lora_alpha;

    /* Active flag */
    int active;

    /* f16→f32 conversion buffers (must be freed on cleanup) */
    float **f16_bufs;
    int     n_f16_bufs;

    /* Tokenizer from GGUF metadata */
    char  **vocab_tokens;   /* token strings, indexed by token id */
    float  *vocab_scores;   /* BPE merge scores per token (SentencePiece) or from merges (GPT-2) */
    int     vocab_scores_size;
    int     vocab_size;     /* number of entries */
    int     bos_id, eos_id; /* special tokens */
    int     add_space_prefix;
    int     is_gpt2_bpe;    /* 1 if tokenizer.ggml.model == "gpt2" */
    int     is_tekken;      /* 1 if tokenizer.ggml.pre == "tekken" */

    /* GPT-2 BPE merges (used to build scores if no native scores) */
    char  **bpe_merges;     /* merge strings "A B" */
    int     n_bpe_merges;

    /* Token hash table for O(1) lookup */
    int    *tok_ht_ids;     /* hash table: token id or -1 */
    int     tok_ht_cap;     /* hash table capacity (power of 2) */

    /* Chat template detection */
    int     chat_style;     /* 0=raw, 1=chatml, 2=llama/mistral [INST], 3=zephyr, 4=phi, 5=gemma, 6=nanollama */

    /* Identity & gamma */
    int     weightless;     /* 1 if no doe_identity.gguf found */
    char    identity_tag[128]; /* doe.identity metadata from GGUF — empty if not DOE's own */
    void   *gamma_data;     /* raw gamma binary blob */
    int     gamma_size;     /* gamma blob size in bytes */
} GGUFIndex;

typedef struct { char name[96]; uint32_t ndim; uint64_t dims[4]; uint32_t dtype; uint64_t offset; } TensorInfo;

static int doe_mul_size(size_t a, size_t b, size_t *out) {
    if (b != 0 && a > SIZE_MAX / b) return 0;
    *out = a * b;
    return 1;
}

static int doe_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (b != 0 && a > UINT64_MAX / b) return 0;
    *out = a * b;
    return 1;
}

static int doe_tensor_elements(const TensorInfo *ti, uint64_t *out) {
    uint64_t n = 1;
    for (uint32_t d = 0; d < ti->ndim; d++) {
        if (ti->dims[d] == 0 || !doe_mul_u64(n, ti->dims[d], &n)) return 0;
    }
    *out = n;
    return 1;
}

static int doe_validate_host_config(GGUFIndex *ps, int require_vocab) {
    if (ps->host_dim <= 0 || ps->host_dim > DOE_MAX_HOST_DIM) {
        fprintf(stderr, "[doe] invalid embedding_length: %d\n", ps->host_dim);
        return 0;
    }
    if (ps->host_n_layers <= 0 || ps->host_n_layers > MAX_LAYERS) {
        fprintf(stderr, "[doe] invalid block_count: %d (max %d)\n", ps->host_n_layers, MAX_LAYERS);
        return 0;
    }
    if (ps->host_heads == 0) ps->host_heads = ps->host_dim / 64;
    if (ps->host_heads <= 0 || ps->host_heads > DOE_MAX_HOST_HEADS) {
        fprintf(stderr, "[doe] invalid head_count: %d\n", ps->host_heads);
        return 0;
    }
    if (ps->host_kv_heads == 0) ps->host_kv_heads = ps->host_heads;
    if (ps->host_kv_heads <= 0 || ps->host_kv_heads > ps->host_heads ||
        ps->host_heads % ps->host_kv_heads != 0) {
        fprintf(stderr, "[doe] invalid head_count_kv: %d for heads=%d\n",
                ps->host_kv_heads, ps->host_heads);
        return 0;
    }
    if (ps->host_head_dim == 0) ps->host_head_dim = ps->host_dim / ps->host_heads;
    if (ps->host_head_dim <= 0 || ps->host_head_dim > DOE_MAX_HOST_HEAD_DIM ||
        ps->host_heads > INT_MAX / ps->host_head_dim ||
        ps->host_kv_heads > INT_MAX / ps->host_head_dim) {
        fprintf(stderr, "[doe] invalid attention head_dim: %d\n", ps->host_head_dim);
        return 0;
    }
    if (ps->host_hidden == 0) {
        if (ps->host_dim > INT_MAX / 4) return 0;
        ps->host_hidden = ps->host_dim * 4;
    }
    if (ps->host_hidden <= 0 || ps->host_hidden > DOE_MAX_HOST_HIDDEN) {
        fprintf(stderr, "[doe] invalid feed_forward_length: %d\n", ps->host_hidden);
        return 0;
    }
    if ((require_vocab && ps->host_vocab <= 0) || ps->host_vocab > DOE_MAX_HOST_VOCAB) {
        fprintf(stderr, "[doe] invalid vocab_size: %d\n", ps->host_vocab);
        return 0;
    }
    return 1;
}

static void doe_print_tensor_shape(const TensorInfo *ti)
{
    fprintf(stderr, "%s ndim=%u dims=[", ti->name, ti->ndim);
    for (uint32_t d = 0; d < ti->ndim; d++)
        fprintf(stderr, "%s%llu", d ? "," : "", (unsigned long long)ti->dims[d]);
    fprintf(stderr, "]");
}

static int doe_tensor_shape1(const TensorInfo *ti, uint64_t d0, const char *role)
{
    if (ti->ndim == 1 && ti->dims[0] == d0) return 1;
    fprintf(stderr, "[doe] tensor shape mismatch for %s (%s): expected [%llu], got ",
            ti->name, role ? role : "tensor", (unsigned long long)d0);
    doe_print_tensor_shape(ti);
    fprintf(stderr, "\n");
    return 0;
}

static int doe_tensor_shape2(const TensorInfo *ti, uint64_t d0, uint64_t d1, const char *role)
{
    if (ti->ndim == 2 && ti->dims[0] == d0 && ti->dims[1] == d1) return 1;
    fprintf(stderr, "[doe] tensor shape mismatch for %s (%s): expected [%llu,%llu], got ",
            ti->name, role ? role : "tensor",
            (unsigned long long)d0, (unsigned long long)d1);
    doe_print_tensor_shape(ti);
    fprintf(stderr, "\n");
    return 0;
}

static int doe_tensor_vocab2(GGUFIndex *ps, const TensorInfo *ti, uint64_t d0, const char *role)
{
    if (ti->ndim != 2 || ti->dims[0] != d0 ||
        ti->dims[1] == 0 || ti->dims[1] > DOE_MAX_HOST_VOCAB ||
        ti->dims[1] > (uint64_t)INT_MAX) {
        fprintf(stderr, "[doe] tensor shape mismatch for %s (%s): expected [%llu,vocab<=%d], got ",
                ti->name, role ? role : "vocab tensor",
                (unsigned long long)d0, DOE_MAX_HOST_VOCAB);
        doe_print_tensor_shape(ti);
        fprintf(stderr, "\n");
        return 0;
    }
    if (ps->host_vocab > 0 && ti->dims[1] != (uint64_t)ps->host_vocab) {
        fprintf(stderr, "[doe] tensor vocab mismatch for %s (%s): metadata vocab=%d, tensor rows=%llu\n",
                ti->name, role ? role : "vocab tensor",
                ps->host_vocab, (unsigned long long)ti->dims[1]);
        return 0;
    }
    if (ps->host_vocab == 0) ps->host_vocab = (int)ti->dims[1];
    return 1;
}

static int doe_host_layer_complete(const GGUFIndex *ps, int l)
{
    int ffn_ok = (ps->host_layers[l].ffn_gate_up && ps->host_layers[l].ffn_down) ||
                 (ps->host_layers[l].ffn_gate && ps->host_layers[l].ffn_up &&
                  ps->host_layers[l].ffn_down);
    return ps->host_layers[l].wq && ps->host_layers[l].wk &&
           ps->host_layers[l].wv && ps->host_layers[l].wo &&
           ps->host_layers[l].attn_norm && ps->host_layers[l].ffn_norm &&
           ffn_ok;
}

static int doe_validate_host_tensor_completeness(const GGUFIndex *ps)
{
    for (int l = 0; l < ps->host_n_layers && l < MAX_LAYERS; l++) {
        if (doe_host_layer_complete(ps, l)) continue;
        fprintf(stderr,
                "[doe] host layer %d incomplete: wq=%d wk=%d wv=%d wo=%d attn_norm=%d ffn_norm=%d gate=%d up=%d gate_up=%d down=%d\n",
                l,
                ps->host_layers[l].wq != NULL,
                ps->host_layers[l].wk != NULL,
                ps->host_layers[l].wv != NULL,
                ps->host_layers[l].wo != NULL,
                ps->host_layers[l].attn_norm != NULL,
                ps->host_layers[l].ffn_norm != NULL,
                ps->host_layers[l].ffn_gate != NULL,
                ps->host_layers[l].ffn_up != NULL,
                ps->host_layers[l].ffn_gate_up != NULL,
                ps->host_layers[l].ffn_down != NULL);
        return 0;
    }
    return 1;
}

static void doe_free_tokenizer_metadata(GGUFIndex *ps) {
    if (ps->vocab_tokens) {
        for (int i = 0; i < ps->vocab_size; i++) free(ps->vocab_tokens[i]);
        free(ps->vocab_tokens);
        ps->vocab_tokens = NULL;
    }
    ps->vocab_size = 0;
    free(ps->vocab_scores);
    ps->vocab_scores = NULL;
    ps->vocab_scores_size = 0;
    if (ps->bpe_merges) {
        for (int i = 0; i < ps->n_bpe_merges; i++) free(ps->bpe_merges[i]);
        free(ps->bpe_merges);
        ps->bpe_merges = NULL;
    }
    ps->n_bpe_merges = 0;
    free(ps->tok_ht_ids);
    ps->tok_ht_ids = NULL;
    ps->tok_ht_cap = 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * ENVIRONMENT SCANNER — DOE opens its eyes
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    char path[256]; char arch[64]; int n_layers, dim, n_heads;
    int64_t file_size; float compatibility;
} DiscoveredGGUF;

typedef struct {
    DiscoveredGGUF ggufs[32]; int n_ggufs;
    int64_t disk_free, mem_available;
    int cpu_count, has_compiler, has_curl;
    char self_path[256];
} Environment;

static int gguf_sniff_skip(FILE *f, uint64_t n) {
    return n <= (uint64_t)LONG_MAX && fseek(f, (long)n, SEEK_CUR) == 0;
}

static int gguf_sniff_skip_value(FILE *f, uint32_t vtype) {
    switch (vtype) {
    case 0: case 1: case 7:
        return gguf_sniff_skip(f, 1);
    case 2: case 3:
        return gguf_sniff_skip(f, 2);
    case 4: case 5: case 6:
        return gguf_sniff_skip(f, 4);
    case 10: case 11: case 12:
        return gguf_sniff_skip(f, 8);
    case 8: {
        uint64_t vlen;
        return fread(&vlen, 8, 1, f) == 1 && gguf_sniff_skip(f, vlen);
    }
    case 9: {
        uint32_t atype;
        uint64_t alen;
        if (fread(&atype, 4, 1, f) != 1 || fread(&alen, 8, 1, f) != 1) return 0;
        size_t esz = 0;
        if (atype == 0 || atype == 1 || atype == 7) esz = 1;
        else if (atype == 2 || atype == 3) esz = 2;
        else if (atype == 4 || atype == 5 || atype == 6) esz = 4;
        else if (atype == 10 || atype == 11 || atype == 12) esz = 8;
        else if (atype == 8) {
            for (uint64_t ai = 0; ai < alen; ai++) {
                uint64_t slen;
                if (fread(&slen, 8, 1, f) != 1 || !gguf_sniff_skip(f, slen)) return 0;
            }
            return 1;
        } else {
            return 0;
        }
        uint64_t bytes = 0;
        return doe_mul_u64(alen, (uint64_t)esz, &bytes) && gguf_sniff_skip(f, bytes);
    }
    default:
        return 0;
    }
}

static int gguf_sniff(const char *path, DiscoveredGGUF *out) {
    if (!path || !out) return 0;
    errno = 0;
    FILE *f = fopen(path, "rb");
    if (!f) {
        int open_errno = errno;
        fprintf(stderr, "[env] cannot open GGUF candidate %s%s%s\n",
                path, open_errno ? ": " : "", open_errno ? strerror(open_errno) : "");
        return 0;
    }
    struct stat st;
    errno = 0;
    if (fstat(fileno(f), &st) != 0) {
        int stat_errno = errno;
        fprintf(stderr, "[env] cannot stat GGUF candidate %s%s%s\n",
                path, stat_errno ? ": " : "", stat_errno ? strerror(stat_errno) : "");
        fclose(f);
        return 0;
    }
    if (st.st_size <= 0 || (uintmax_t)st.st_size > (uintmax_t)INT64_MAX) {
        fprintf(stderr, "[env] invalid GGUF candidate size for %s: %lld bytes\n",
                path, (long long)st.st_size);
        fclose(f);
        return 0;
    }
    out->file_size = (int64_t)st.st_size;
    snprintf(out->path, 256, "%s", path);
    memset(out->arch, 0, 64); out->n_layers = 0; out->dim = 0; out->n_heads = 0;
    uint32_t magic; if (fread(&magic, 4, 1, f) != 1 || magic != 0x46554747) { fclose(f); return 0; }
    uint32_t version;
    uint64_t n_tensors, n_kv;
    if (fread(&version, 4, 1, f) != 1 ||
        fread(&n_tensors, 8, 1, f) != 1 ||
        fread(&n_kv, 8, 1, f) != 1) { fclose(f); return 0; }
    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t klen; if (fread(&klen, 8, 1, f) != 1) { fclose(f); return 0; }
        if (klen > 255) {
            uint32_t vtype;
            if (!gguf_sniff_skip(f, klen) || fread(&vtype, 4, 1, f) != 1 ||
                !gguf_sniff_skip_value(f, vtype)) { fclose(f); return 0; }
            continue;
        }
        char key[256];
        if (fread(key, 1, klen, f) != klen) { fclose(f); return 0; }
        key[klen] = '\0';
        uint32_t vtype; if (fread(&vtype, 4, 1, f) != 1) { fclose(f); return 0; }
        if (vtype == 8) { /* string */
            uint64_t vlen;
            char val[256];
            if (fread(&vlen, 8, 1, f) != 1) { fclose(f); return 0; }
            int rl = vlen < 255 ? (int)vlen : 255;
            if (fread(val, 1, rl, f) != (size_t)rl) { fclose(f); return 0; }
            val[rl] = '\0';
            if (vlen > 255 && !gguf_sniff_skip(f, vlen - 255)) { fclose(f); return 0; }
            if (strstr(key, "general.architecture")) snprintf(out->arch, 64, "%s", val);
        } else if (vtype == 4) {
            uint32_t val;
            if (fread(&val, 4, 1, f) != 1) { fclose(f); return 0; }
            if (strstr(key, "embedding_length")) out->dim = (int)val;
            else if (strstr(key, "block_count")) out->n_layers = (int)val;
            else if (strstr(key, "head_count") && !strstr(key, "kv")) out->n_heads = (int)val;
        } else if (!gguf_sniff_skip_value(f, vtype)) { fclose(f); return 0; }
    }
    fclose(f);
    return (out->arch[0] != '\0' && out->dim > 0);
}

static int env_path_has_exec(const char *name) {
    if (!name || !*name) return 0;
    if (strchr(name, '/')) return access(name, X_OK) == 0;
    const char *path = getenv("PATH");
    if (!path || !*path) path = "/bin:/usr/bin:/usr/local/bin:/opt/homebrew/bin";
    size_t name_len = strlen(name);
    const char *p = path;
    while (p && *p) {
        const char *colon = strchr(p, ':');
        size_t dir_len = colon ? (size_t)(colon - p) : strlen(p);
        char full[512];
        int ok = 0;
        if (dir_len == 0) {
            ok = snprintf(full, sizeof(full), "./%s", name);
        } else if (dir_len + 1 + name_len < sizeof(full)) {
            memcpy(full, p, dir_len);
            full[dir_len] = '/';
            memcpy(full + dir_len + 1, name, name_len + 1);
            ok = (int)(dir_len + 1 + name_len);
        }
        if (ok > 0 && ok < (int)sizeof(full) && access(full, X_OK) == 0) return 1;
        p = colon ? colon + 1 : NULL;
    }
    return 0;
}

static int env_has_suffix(const char *s, const char *suffix) {
    if (!s || !suffix) return 0;
    size_t slen = strlen(s), tlen = strlen(suffix);
    return slen >= tlen && strcmp(s + slen - tlen, suffix) == 0;
}

static void env_scan_ggufs(Environment *env, const char *dir_path, int depth_remaining) {
    if (!env || !dir_path || depth_remaining < 0 || env->n_ggufs >= 32) return;
    DIR *dir = opendir(dir_path);
    if (!dir) return;
    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL && env->n_ggufs < 32) {
        const char *name = ent->d_name;
        if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0) continue;
        char path[256];
        int npath = snprintf(path, sizeof(path), "%s/%s", dir_path, name);
        if (npath < 0 || npath >= (int)sizeof(path)) continue;
        if (env_has_suffix(name, ".gguf")) {
            DiscoveredGGUF dg;
            if (gguf_sniff(path, &dg)) env->ggufs[env->n_ggufs++] = dg;
            continue;
        }
        if (depth_remaining > 0) {
            struct stat st;
            if (lstat(path, &st) == 0 && S_ISDIR(st.st_mode))
                env_scan_ggufs(env, path, depth_remaining - 1);
        }
    }
    closedir(dir);
}

static void env_scan(Environment *env, const char *self_src) {
    memset(env, 0, sizeof(Environment));
    snprintf(env->self_path, 256, "%s", self_src);
    env->cpu_count = (int)sysconf(_SC_NPROCESSORS_ONLN);
#ifdef __linux__
    env->mem_available = (int64_t)sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGESIZE);
    struct statvfs sv; if (statvfs(".", &sv) == 0) env->disk_free = (int64_t)sv.f_bavail * sv.f_frsize;
#elif defined(__APPLE__)
    int64_t mem = 0; size_t len = sizeof(mem);
    sysctlbyname("hw.memsize", &mem, &len, NULL, 0); env->mem_available = mem;
    struct statfs sf; if (statfs(".", &sf) == 0) env->disk_free = (int64_t)sf.f_bavail * sf.f_bsize;
#endif
    env->has_compiler = env_path_has_exec("cc");
    env->has_curl = env_path_has_exec("curl");
    env_scan_ggufs(env, ".", 3);
    printf("[env] cpu=%d mem=%.1fGB disk=%.1fGB compiler=%s curl=%s ggufs=%d\n",
           env->cpu_count, (float)env->mem_available/(1024*1024*1024),
           (float)env->disk_free/(1024*1024*1024),
           env->has_compiler?"yes":"no", env->has_curl?"yes":"no", env->n_ggufs);
    for (int i = 0; i < env->n_ggufs; i++)
        printf("  [gguf] %s arch=%s dim=%d layers=%d %.1fMB\n",
               env->ggufs[i].path, env->ggufs[i].arch, env->ggufs[i].dim,
               env->ggufs[i].n_layers, (float)env->ggufs[i].file_size/(1024*1024));
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * INDEX LOAD — mmap GGUF, wire weight pointers, profile layers, attach LoRA.
 * the weights are substrate. DOE is the architecture.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static int init_lora_expert(LoraExpert *e, int dim, int rank, float freq) {
    size_t lora_elems = 0;
    if (dim <= 0 || rank <= 0 ||
        !doe_mul_size((size_t)dim, (size_t)rank, &lora_elems) ||
        lora_elems > (size_t)INT_MAX) {
        fprintf(stderr, "[doe] invalid LoRA expert shape (dim=%d rank=%d)\n", dim, rank);
        memset(e, 0, sizeof(*e));
        return 0;
    }
    e->lora_A = calloc(lora_elems, sizeof(float));
    e->lora_B = calloc(lora_elems, sizeof(float));
    if (!e->lora_A || !e->lora_B) {
        fprintf(stderr, "[doe] LoRA expert allocation failed (dim=%d rank=%d)\n", dim, rank);
        free(e->lora_A);
        free(e->lora_B);
        memset(e, 0, sizeof(*e));
        return 0;
    }
    float scale = 0.02f / sqrtf((float)rank);
    for (size_t i = 0; i < lora_elems; i++) e->lora_A[i] = rand_normal() * scale;
    for (size_t i = 0; i < lora_elems; i++) e->lora_B[i] = rand_normal() * scale;
    e->frequency = freq;
    e->vitality = 0.7f;
    e->alive = 1;
    e->attention_bias = 0.0f;
    e->layer_focus = 1.0f;
    e->low_vitality_count = 0;
    return 1;
}

static void free_lora_expert(LoraExpert *e) {
    free(e->lora_A); free(e->lora_B);
    memset(e, 0, sizeof(*e));
}

static int field_recount_alive(FieldLayer *fl) {
    int n = 0;
    if (!fl) return 0;
    for (int e = 0; e < MAX_EXPERTS; e++)
        if (fl->experts[e].alive) n++;
    fl->n_alive = n;
    return n;
}

static void field_release_resources(FieldLayer *fl) {
    if (!fl) return;
    int host_layer_idx = fl->host_layer_idx;
    free(fl->parliament.w_vote);
    for (int e = 0; e < MAX_EXPERTS; e++)
        if (fl->experts[e].lora_A || fl->experts[e].lora_B)
            free_lora_expert(&fl->experts[e]);
    memset(fl, 0, sizeof(*fl));
    fl->host_layer_idx = host_layer_idx;
}

static float sanitize_expert_vitality(float v, int layer, int expert) {
    if (!isfinite(v)) {
        printf("[mycelium] invalid vitality in spore L%d e%d; resetting to 0.5\n", layer, expert);
        return 0.5f;
    }
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

static float sanitize_expert_frequency(float f, int layer, int expert) {
    const float tau = 6.2831853f;
    if (!isfinite(f)) {
        printf("[mycelium] invalid frequency in spore L%d e%d; resetting\n", layer, expert);
        return tau * (float)expert / (float)MAX_EXPERTS;
    }
    f = fmodf(f, tau);
    if (!isfinite(f)) return tau * (float)expert / (float)MAX_EXPERTS;
    if (f < 0.0f) f += tau;
    return f;
}

#if defined(__GNUC__) || defined(__clang__)
#define DOE_UNUSED_FN __attribute__((unused))
#else
#define DOE_UNUSED_FN
#endif

static float DOE_UNUSED_FN field_frequency_delta(float a, float b) {
    const float tau = 6.2831853f;
    if (!isfinite(a) || !isfinite(b)) return INFINITY;
    float d = fabsf(a - b);
    d = fmodf(d, tau);
    if (!isfinite(d)) return INFINITY;
    return d > tau * 0.5f ? tau - d : d;
}

static int DOE_UNUSED_FN field_resonance_frequencies_compatible(const GGUFIndex *ps, int *out_layer, int *out_expert) {
    if (!ps) return 0;
    for (int l = 1; l < ps->n_field_layers; l++) {
        for (int e = 0; e < MAX_EXPERTS; e++) {
            if (!ps->field_layers[0].experts[e].alive ||
                !ps->field_layers[l].experts[e].alive)
                continue;
            if (field_frequency_delta(ps->field_layers[0].experts[e].frequency,
                                      ps->field_layers[l].experts[e].frequency) > 1e-6f) {
                if (out_layer) *out_layer = l;
                if (out_expert) *out_expert = e;
                return 0;
            }
        }
    }
    return 1;
}
#undef DOE_UNUSED_FN

#ifdef USE_METAL
/* Parliament-on-GPU step 1 — pack the (inference-frozen) per-layer vote rows +
 * expert LoRA matrices into one contiguous, page-aligned arena and register it
 * as a zero-copy GPU region. The election/inject kernels then bind w_vote / A /
 * B by offset (resolve_region) instead of uploading per token. Built once at
 * load, only when Metal is live and the parliament is active (alpha != 0). The
 * experts are static at --train 0 (births/deaths gated on g_train, :3511), so
 * one pack suffices — no re-sync. Dead expert slots stay zero (memset arena) so
 * a gate weight of 0 over them is a true no-op.
 *   Per layer: [ w_vote MAX_EXPERTS*D | per-expert (B rank*D, A D*rank) x MAX_EXPERTS ]. */
static void field_repack_arena(GGUFIndex *ps) {
    if (ps->expert_arena_ready) return;
    if (g_train) {
        fprintf(stderr, "[doe] parliament arena disabled under --train 1; CPU path keeps live experts fresh\n");
        return;
    }
    int D = ps->host_dim, rank = ps->lora_rank, L = ps->n_field_layers;
    for (int l = 1; l < L; l++) {
        for (int e = 0; e < MAX_EXPERTS; e++) {
            if (!!ps->field_layers[l].experts[e].alive != !!ps->field_layers[0].experts[e].alive) {
                fprintf(stderr, "[doe] parliament arena disabled: alive mask differs at layer %d expert %d\n", l, e);
                return;
            }
        }
    }
    int freq_layer = -1, freq_expert = -1;
    if (!field_resonance_frequencies_compatible(ps, &freq_layer, &freq_expert)) {
        fprintf(stderr, "[doe] parliament arena disabled: expert frequency differs at layer %d expert %d\n",
                freq_layer, freq_expert);
        return;
    }
    size_t lora_b = 0, lora_a = 0, per_expert = 0, vote_floats = 0;
    size_t expert_floats = 0, layer_floats = 0, total_floats = 0, raw_bytes = 0;
    if (D <= 0 || rank <= 0 || L <= 0 ||
        !doe_mul_size((size_t)rank, (size_t)D, &lora_b) ||
        !doe_mul_size((size_t)D, (size_t)rank, &lora_a) ||
        lora_b > SIZE_MAX - lora_a ||
        !doe_mul_size((size_t)MAX_EXPERTS, (size_t)D, &vote_floats) ||
        !doe_mul_size((size_t)MAX_EXPERTS, lora_b + lora_a, &expert_floats) ||
        vote_floats > SIZE_MAX - expert_floats ||
        !doe_mul_size(vote_floats + expert_floats, (size_t)L, &total_floats) ||
        !doe_mul_size(total_floats, sizeof(float), &raw_bytes)) {
        fprintf(stderr, "[doe] parliament arena disabled: invalid tensor shape (layers=%d dim=%d rank=%d)\n",
                L, D, rank);
        return;
    }
    per_expert = lora_b + lora_a;                                       /* B then A */
    layer_floats = vote_floats + expert_floats;
    int page_size = getpagesize();
    if (page_size <= 0) {
        fprintf(stderr, "[doe] parliament arena disabled: invalid page size %d\n", page_size);
        return;
    }
    size_t pg = (size_t)page_size;
    if (raw_bytes > SIZE_MAX - (pg - 1)) {
        fprintf(stderr, "[doe] parliament arena disabled: byte size overflow\n");
        return;
    }
    size_t bytes = (raw_bytes + pg - 1) / pg * pg;                      /* register needs page-aligned len */
    float *arena = NULL;
    if (posix_memalign((void**)&arena, pg, bytes) != 0 || !arena) {
        fprintf(stderr, "[doe] parliament arena posix_memalign(%zu) failed — parliament stays on CPU\n", bytes);
        return;
    }
    memset(arena, 0, bytes);                                            /* dead experts -> zero contribution */
    for (int l = 0; l < L; l++) {
        FieldLayer *fl = &ps->field_layers[l];
        float *base = arena + (size_t)l * layer_floats;
        memcpy(base, fl->parliament.w_vote, (size_t)MAX_EXPERTS * D * sizeof(float));
        float *ep = base + (size_t)MAX_EXPERTS * D;
        for (int e = 0; e < MAX_EXPERTS; e++) {
            if (!fl->experts[e].alive) continue;                        /* leave zero */
            float *eb = ep + (size_t)e * per_expert;
            memcpy(eb,                    fl->experts[e].lora_B, (size_t)rank * D * sizeof(float)); /* B [rank,D] */
            memcpy(eb + (size_t)rank * D, fl->experts[e].lora_A, (size_t)D * rank * sizeof(float)); /* A [D,rank] */
        }
    }
    if (nt_metal_register_region(arena, bytes) != 0) {
        fprintf(stderr, "[doe] parliament arena register_region failed — parliament stays on CPU\n");
        free(arena);
        return;
    }
    ps->expert_arena        = arena;
    ps->expert_arena_floats = total_floats;
    ps->expert_layer_floats = layer_floats;
    ps->expert_arena_ready  = 1;
    fprintf(stderr, "[doe] parliament arena: %.1f MB registered (%d layers x %zu f/layer; arena[0]=%.6f src=%.6f)\n",
            (double)bytes / (1024.0*1024.0), L, layer_floats,
            arena[0], ps->field_layers[0].parliament.w_vote[0]);
}
#endif

static int tok_lookup(GGUFIndex *ps, const char *s, int len);
static void tok_ht_build(GGUFIndex *ps);
static void build_gpt2_scores(GGUFIndex *ps);

static int index_load(GGUFIndex *ps, const char *path) {
    memset(ps, 0, sizeof(GGUFIndex));
    snprintf(ps->host_path, 256, "%s", path);
    ps->lora_rank = LORA_RANK;
    ps->lora_alpha = F.lora_alpha;
    ps->bos_id = 1; ps->eos_id = 2; /* defaults, overridden by GGUF */
    ps->rope_theta = 10000.0f;
    ps->rms_norm_eps = 1e-5f;
    ps->add_space_prefix = 1;

    errno = 0;
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        int open_errno = errno;
        printf("[doe] cannot open %s%s%s\n", path,
               open_errno ? ": " : "", open_errno ? strerror(open_errno) : "");
        return 0;
    }
    struct stat st;
    errno = 0;
    if (fstat(fd, &st) != 0) {
        int stat_errno = errno;
        printf("[doe] cannot stat %s%s%s\n", path,
               stat_errno ? ": " : "", stat_errno ? strerror(stat_errno) : "");
        close(fd);
        return 0;
    }
    if (st.st_size <= 0 || (uintmax_t)st.st_size > (uintmax_t)SIZE_MAX) {
        printf("[doe] invalid GGUF file size for %s: %lld bytes\n",
               path, (long long)st.st_size);
        close(fd);
        return 0;
    }
    ps->mmap_size = (size_t)st.st_size;
    errno = 0;
    ps->mmap_base = mmap(NULL, ps->mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);
    int mmap_errno = errno;
    close(fd);
    if (ps->mmap_base == MAP_FAILED) {
        printf("[doe] cannot mmap %s (%zu bytes)%s%s\n",
               path, ps->mmap_size,
               mmap_errno ? ": " : "", mmap_errno ? strerror(mmap_errno) : "");
        ps->mmap_base = NULL;
        ps->mmap_size = 0;
        return 0;
    }
#ifdef USE_METAL
    /* Metal: register the whole packed GGUF block as a resident zero-copy GPU buffer (Phase 2 —
     * weights bind by offset, no per-token upload). mmap base is page-aligned;
     * register_base page-rounds the exact file length for Metal NoCopy. */
    if (nt_metal_available()) nt_metal_register_base(ps->mmap_base, ps->mmap_size);
#endif

    TensorInfo *tinfo = NULL;

    /* Parse GGUF header */
    uint8_t *p = ps->mmap_base, *pend = ps->mmap_base + ps->mmap_size;
    #define PC(n) do { uint64_t _need = (uint64_t)(n); if (_need > (uint64_t)(pend - p)) goto bail; } while(0)
    PC(4); uint32_t magic = *(uint32_t*)p; p += 4;
    if (magic != 0x46554747) goto bail;
    PC(4); p += 4; /* version */
    PC(8); uint64_t n_tensors = *(uint64_t*)p; p += 8;
    PC(8); uint64_t n_kv = *(uint64_t*)p; p += 8;

    for (uint64_t i = 0; i < n_kv; i++) {
        PC(8); uint64_t klen = *(uint64_t*)p; p += 8;
        char key[256];
        if (klen > 255) {
            /* D-M1: an oversized key still has a vtype + value after it — skip
             * only the key bytes and let the vtype handling below parse/skip the
             * value, or the parser desyncs. Empty key matches no strstr. */
            PC(klen); p += klen; key[0] = '\0';
        } else {
            PC(klen); memcpy(key, p, klen); key[klen] = '\0'; p += klen; /* D-M2: bound key payload */
        }
        PC(4); uint32_t vtype = *(uint32_t*)p; p += 4;
        if (vtype == 8) { /* string */
            PC(8); uint64_t vlen = *(uint64_t*)p; p += 8;
            PC(vlen); /* D-M2: bound the string payload before any memcpy reads it */
            if (strstr(key, "general.architecture") && vlen < 64) {
                memcpy(ps->host_arch, p, vlen); ps->host_arch[vlen] = 0;
            }
            if (strstr(key, "tokenizer.ggml.model") && vlen < 20) {
                char tok_model[24]; memcpy(tok_model, p, vlen); tok_model[vlen] = 0;
                if (strcmp(tok_model, "gpt2") == 0) ps->is_gpt2_bpe = 1;
            }
            if (strstr(key, "tokenizer.ggml.pre") && vlen < 64) {
                char tok_pre[64]; memcpy(tok_pre, p, vlen); tok_pre[vlen] = 0;
                if (strcmp(tok_pre, "tekken") == 0) ps->is_tekken = 1;
            }
            /* DOE identity fingerprint — this GGUF is DOE's own */
            if (strcmp(key, "doe.identity") == 0 && vlen < 128) {
                memcpy(ps->identity_tag, p, vlen); ps->identity_tag[vlen] = 0;
                printf("[identity] GGUF self-identifies: \"%s\"\n", ps->identity_tag);
            }
            /* Detect chat template style from template string */
            if (strstr(key, "chat_template") && vlen > 10 && vlen < 100000) {
                /* Search for distinctive patterns in the Jinja template */
                char *tmpl = malloc(vlen + 1);
                if (!tmpl) goto bail;
                memcpy(tmpl, p, vlen); tmpl[vlen] = 0;
                if (strstr(tmpl, "im_start"))       ps->chat_style = 1; /* ChatML */
                else if (strstr(tmpl, "[INST]"))     ps->chat_style = 2; /* Llama/Mistral */
                else if (strstr(tmpl, "<|user|>"))   ps->chat_style = 3; /* Zephyr */
                else if (strstr(tmpl, "<|end|>"))    ps->chat_style = 4; /* Phi */
                else if (strstr(tmpl, "start_of_turn")) ps->chat_style = 5; /* Gemma */
                free(tmpl);
            }
            p += vlen;
        } else if (vtype == 4) { /* uint32 */
            PC(4); uint32_t val = *(uint32_t*)p; p += 4;
            if (strstr(key, "embedding_length")) ps->host_dim = (int)val;
            else if (strstr(key, "block_count")) ps->host_n_layers = (int)val;
            else if (strstr(key, "head_count") && !strstr(key, "kv")) ps->host_heads = (int)val;
            else if (strstr(key, "head_count_kv")) ps->host_kv_heads = (int)val;
            else if (strstr(key, "feed_forward_length")) ps->host_hidden = (int)val;
            else if (strstr(key, "vocab_size")) ps->host_vocab = (int)val;
            else if (strstr(key, "bos_token_id")) ps->bos_id = (int)val;
            else if (strstr(key, "eos_token_id")) ps->eos_id = (int)val;
            else if (strstr(key, "add_space_prefix")) ps->add_space_prefix = (int)val;
            else if (strstr(key, "attention.key_length")) ps->host_head_dim = (int)val;
        } else if (vtype == 6) { /* float32 */
            PC(4); float fval; memcpy(&fval, p, 4); p += 4;
            if (strstr(key, "rope.freq_base")) ps->rope_theta = fval;
            else if (strstr(key, "layer_norm_rms_epsilon")) ps->rms_norm_eps = fval;
        } else if (vtype == 0 || vtype == 7) {
            PC(1); uint8_t bval = *p; p += 1;
            if (strstr(key, "add_space_prefix")) ps->add_space_prefix = bval;
        } else if (vtype == 1) { PC(1); p += 1; }              /* int8 */
        else if (vtype == 2 || vtype == 3) { PC(2); p += 2; }  /* uint16, int16 */
        else if (vtype == 5) { PC(4); p += 4; }                /* int32 */
        else if (vtype == 10 || vtype == 11 || vtype == 12) { PC(8); p += 8; } /* uint64, int64, float64 */
        else if (vtype == 9) { /* array */
            PC(4); uint32_t atype = *(uint32_t*)p; p += 4;
            PC(8); uint64_t alen = *(uint64_t*)p; p += 8;
            size_t elem_sz = 0;
            if (atype == 0 || atype == 1 || atype == 7) elem_sz = 1;
            else if (atype == 2 || atype == 3) elem_sz = 2;
            else if (atype == 4 || atype == 5 || atype == 6) {
                elem_sz = 4;
                /* float32 array: tokenizer.ggml.scores */
                if (atype == 6 && strstr(key, "tokenizer.ggml.scores")) {
                    uint64_t score_bytes = 0;
                    if (alen > DOE_MAX_HOST_VOCAB ||
                        alen > (uint64_t)SIZE_MAX / sizeof(float) ||
                        ps->vocab_scores ||
                        !doe_mul_u64(alen, 4, &score_bytes)) {
                        fprintf(stderr, "[doe] malformed tokenizer scores array len=%llu key=%s\n",
                                (unsigned long long)alen, key);
                        goto bail;
                    }
                    PC(score_bytes); /* D-M2: bound the scores payload before memcpy */
                    ps->vocab_scores = malloc((alen ? (size_t)alen : 1) * sizeof(float));
                    if (!ps->vocab_scores) goto bail;
                    memcpy(ps->vocab_scores, p, (size_t)score_bytes);
                    ps->vocab_scores_size = (int)alen;
                }
            }
            else if (atype == 10 || atype == 11 || atype == 12) elem_sz = 8;
            else if (atype == 8) {
                /* array of strings */
                int is_vocab = strstr(key, "tokenizer.ggml.tokens") != NULL;
                int is_merges = strstr(key, "tokenizer.ggml.merges") != NULL;
                if (is_vocab) {
                    if (alen > DOE_MAX_HOST_VOCAB ||
                        alen > (uint64_t)INT_MAX ||
                        alen > (uint64_t)SIZE_MAX / sizeof(char*) ||
                        ps->vocab_tokens) {
                        fprintf(stderr, "[doe] malformed tokenizer vocab array len=%llu key=%s\n",
                                (unsigned long long)alen, key);
                        goto bail;
                    }
                    ps->vocab_tokens = calloc(alen ? (size_t)alen : 1, sizeof(char*));
                    if (!ps->vocab_tokens) goto bail;
                    ps->vocab_size = (int)alen;
                }
                if (is_merges) {
                    if (alen > 500000 ||
                        alen > (uint64_t)INT_MAX ||
                        alen > (uint64_t)SIZE_MAX / sizeof(char*) ||
                        ps->bpe_merges) {
                        fprintf(stderr, "[doe] malformed tokenizer merges array len=%llu key=%s\n",
                                (unsigned long long)alen, key);
                        goto bail;
                    }
                    ps->bpe_merges = calloc(alen ? (size_t)alen : 1, sizeof(char*));
                    if (!ps->bpe_merges) goto bail;
                    ps->n_bpe_merges = (int)alen;
                }
                for (uint64_t ai = 0; ai < alen; ai++) {
                    PC(8); uint64_t slen = *(uint64_t*)p; p += 8;
                    if (slen > 1000000 || (uint64_t)(pend - p) < slen) goto bail; /* sanity */
                    if (is_vocab && ps->vocab_tokens && ai < (uint64_t)ps->vocab_size) {
                        ps->vocab_tokens[ai] = malloc(slen + 1);
                        if (!ps->vocab_tokens[ai]) goto bail;
                        memcpy(ps->vocab_tokens[ai], p, slen);
                        ps->vocab_tokens[ai][slen] = '\0';
                    }
                    if (is_merges && ps->bpe_merges && ai < (uint64_t)ps->n_bpe_merges) {
                        ps->bpe_merges[ai] = malloc(slen + 1);
                        if (!ps->bpe_merges[ai]) goto bail;
                        memcpy(ps->bpe_merges[ai], p, slen);
                        ps->bpe_merges[ai][slen] = '\0';
                    }
                    p += slen;
                }
                continue;
            }
            else {
                fprintf(stderr, "[doe] unknown GGUF array metadata type %u for key %s\n", atype, key);
                goto bail;
            }
            uint64_t arr_bytes = 0;
            if (!doe_mul_u64(alen, (uint64_t)elem_sz, &arr_bytes)) goto bail;
            PC(arr_bytes); /* D-M2: bound the array payload before skipping past it */
            p += arr_bytes;
        } else {
            fprintf(stderr, "[doe] unknown GGUF metadata value type %u for key %s\n", vtype, key);
            goto bail;
        }
    }
    if (ps->vocab_tokens && ps->vocab_size <= 0) {
        fprintf(stderr, "[doe] tokenizer vocab is empty\n");
        goto bail;
    }
    if (ps->vocab_scores && ps->vocab_tokens && ps->vocab_scores_size != ps->vocab_size) {
        fprintf(stderr, "[doe] tokenizer scores length %d does not match vocab length %d\n",
                ps->vocab_scores_size, ps->vocab_size);
        goto bail;
    }
    if (!doe_validate_host_config(ps, 0)) goto bail; /* D-L8/Fable F-1: bound host dimensions before allocation/math */

    /* Parse tensor info */
    if (n_tensors > 20000) goto bail;
    tinfo = calloc(n_tensors, sizeof(TensorInfo));
    if (!tinfo) goto bail;
    for (uint64_t i = 0; i < n_tensors; i++) {
        PC(8); uint64_t nlen = *(uint64_t*)p; p += 8;
        if (nlen > 256) goto bail;
        int nl = nlen < 95 ? (int)nlen : 95;
        PC(nlen); memcpy(tinfo[i].name, p, nl); tinfo[i].name[nl] = '\0'; p += nlen;
        PC(4); tinfo[i].ndim = *(uint32_t*)p; p += 4;
        if (tinfo[i].ndim > 4) goto bail;
        for (uint32_t d = 0; d < tinfo[i].ndim; d++) { PC(8); tinfo[i].dims[d] = *(uint64_t*)p; p += 8; }
        PC(4); tinfo[i].dtype = *(uint32_t*)p; p += 4;
        PC(8); tinfo[i].offset = *(uint64_t*)p; p += 8;
    }

    uint64_t header_size = p - ps->mmap_base;
    uint64_t data_start = ((header_size + 31) / 32) * 32;

    /* dequantized f32 buffers — tracked in GGUFIndex for cleanup */
    ps->f16_bufs = NULL; ps->n_f16_bufs = 0;

    /* Wire weight pointers — supports f32, f16, Q4_0, Q8_0, Q4_K, Q6_K */
    int wired = 0;
    uint64_t host_d = (uint64_t)ps->host_dim;
    uint64_t host_h = (uint64_t)ps->host_hidden;
    uint64_t q_dim = (uint64_t)ps->host_heads * (uint64_t)ps->host_head_dim;
    uint64_t kv_dim = (uint64_t)ps->host_kv_heads * (uint64_t)ps->host_head_dim;
    for (uint64_t i = 0; i < n_tensors; i++) {
        uint32_t dt = tinfo[i].dtype;
        if (dt != 0 && dt != 1 && dt != 2 && dt != 6 && dt != 8 && dt != 12 && dt != 14) continue;
        uint64_t n_elems = 0;
        if (!doe_tensor_elements(&tinfo[i], &n_elems)) {
            printf("[doe] WARNING: tensor %s has invalid dimensions, skipping\n", tinfo[i].name);
            continue;
        }
        uint64_t raw_bytes = quant_raw_bytes(dt, n_elems);
        if (data_start > ps->mmap_size ||
            tinfo[i].offset > ps->mmap_size - data_start ||
            raw_bytes == 0) {
            if (raw_bytes > 0)
                printf("[doe] WARNING: tensor %s OOB offset (%lu+%lu > %lu), skipping\n",
                       tinfo[i].name, (unsigned long)data_start, (unsigned long)tinfo[i].offset,
                       (unsigned long)ps->mmap_size);
            continue;
        }
        uint64_t byte_offset = data_start + tinfo[i].offset;
        if (raw_bytes > ps->mmap_size - byte_offset) {
            printf("[doe] WARNING: tensor %s OOB (%lu+%lu > %lu), skipping\n",
                   tinfo[i].name, (unsigned long)byte_offset, (unsigned long)raw_bytes,
                   (unsigned long)ps->mmap_size);
            continue;
        }
        float *data; int data_dt = 0;
        const uint8_t *src = ps->mmap_base + byte_offset;
        char *n = tinfo[i].name;
        int layer_idx = -1;
        int is_layer_tensor = (sscanf(n, "blk.%d.", &layer_idx) == 1 &&
                               layer_idx >= 0 && layer_idx < MAX_LAYERS &&
                               layer_idx < ps->host_n_layers);
        if (strcmp(n, "token_embd.weight") == 0) {
            if (!doe_tensor_vocab2(ps, &tinfo[i], host_d, "token embedding")) goto bail;
        } else if (strcmp(n, "output_norm.weight") == 0) {
            if (!doe_tensor_shape1(&tinfo[i], host_d, "output norm")) goto bail;
        } else if (strcmp(n, "output.weight") == 0) {
            if (!doe_tensor_vocab2(ps, &tinfo[i], host_d, "lm head")) goto bail;
        } else if (is_layer_tensor) {
            if (strstr(n, "attn_q.weight")) {
                if (!doe_tensor_shape2(&tinfo[i], host_d, q_dim, "attention q")) goto bail;
            } else if (strstr(n, "attn_k.weight")) {
                if (!doe_tensor_shape2(&tinfo[i], host_d, kv_dim, "attention k")) goto bail;
            } else if (strstr(n, "attn_v.weight")) {
                if (!doe_tensor_shape2(&tinfo[i], host_d, kv_dim, "attention v")) goto bail;
            } else if (strstr(n, "attn_output.weight")) {
                if (!doe_tensor_shape2(&tinfo[i], q_dim, host_d, "attention output")) goto bail;
            } else if (strstr(n, "attn_q.bias")) {
                if (!doe_tensor_shape1(&tinfo[i], q_dim, "attention q bias")) goto bail;
            } else if (strstr(n, "attn_k.bias")) {
                if (!doe_tensor_shape1(&tinfo[i], kv_dim, "attention k bias")) goto bail;
            } else if (strstr(n, "attn_v.bias")) {
                if (!doe_tensor_shape1(&tinfo[i], kv_dim, "attention v bias")) goto bail;
            } else if (strstr(n, "ffn_gate.weight") && !strstr(n, "ffn_gate_inp") && !strstr(n, "ffn_gate_up")) {
                if (!doe_tensor_shape2(&tinfo[i], host_d, host_h, "ffn gate")) goto bail;
            } else if (strstr(n, "ffn_up.weight") && !strstr(n, "gate_up")) {
                if (tinfo[i].ndim != 2 || tinfo[i].dims[0] != host_d ||
                    (tinfo[i].dims[1] != host_h && tinfo[i].dims[1] != host_h * 2u)) {
                    fprintf(stderr, "[doe] tensor shape mismatch for %s (ffn up/gate_up): expected [%llu,%llu] or [%llu,%llu], got ",
                            n,
                            (unsigned long long)host_d, (unsigned long long)host_h,
                            (unsigned long long)host_d, (unsigned long long)(host_h * 2u));
                    doe_print_tensor_shape(&tinfo[i]);
                    fprintf(stderr, "\n");
                    goto bail;
                }
            } else if (strstr(n, "ffn_down.weight")) {
                if (!doe_tensor_shape2(&tinfo[i], host_h, host_d, "ffn down")) goto bail;
            } else if (strstr(n, "ffn_gate_up_proj") || strstr(n, "ffn_gate_up.weight")) {
                if (!doe_tensor_shape2(&tinfo[i], host_d, host_h * 2u, "fused ffn gate_up")) goto bail;
            } else if (strstr(n, "attn_norm.weight")) {
                if (!doe_tensor_shape1(&tinfo[i], host_d, "attention norm")) goto bail;
            } else if (strstr(n, "ffn_norm.weight")) {
                if (!doe_tensor_shape1(&tinfo[i], host_d, "ffn norm")) goto bail;
            }
        }
        /* matvec weights stay PACKED (dequantized inline by doe_qmatvec, no f32
         * blow-up); embeddings / norms / biases dequant to f32 as before. */
        int is_mv_w = strstr(n,"attn_q.weight")||strstr(n,"attn_k.weight")||strstr(n,"attn_v.weight")||
                      strstr(n,"attn_output.weight")||strstr(n,"ffn_gate.weight")||strstr(n,"ffn_up.weight")||
                      strstr(n,"ffn_down.weight")||strstr(n,"ffn_gate_up")||strcmp(n,"output.weight")==0;
        if (dt == 0) {
            data = (float*)src; /* f32: point directly into mmap */
        } else if (is_mv_w && pq_for(dt, (int)tinfo[i].dims[0])) {
            data = (float*)src; /* keep packed — matvec dequants inline */
            data_dt = dt;
        } else {
            /* dequantize to f32 */
            if (n_elems > SIZE_MAX / sizeof(float)) {
                printf("[doe] WARNING: tensor %s too large to dequantize, skipping\n", n);
                continue;
            }
            data = malloc(n_elems * sizeof(float));
            if (!data) {
                printf("[doe] WARNING: tensor %s dequant allocation failed, skipping\n", n);
                continue;
            }
            if (dt == 1) { /* f16 */
                const uint16_t *h = (const uint16_t*)src;
                for (uint64_t j = 0; j < n_elems; j++) data[j] = f16_to_f32(h[j]);
            } else if (dt == 2) dequant_q4_0(src, data, n_elems);
            else if (dt == 6) dequant_q5_0(src, data, n_elems);
            else if (dt == 8) dequant_q8_0(src, data, n_elems);
            else if (dt == 12) dequant_q4_k(src, data, n_elems);
            else if (dt == 14) dequant_q6_k(src, data, n_elems);
            float **new_bufs = realloc(ps->f16_bufs, (ps->n_f16_bufs+1)*sizeof(float*));
            if (!new_bufs) {
                free(data);
                goto bail;
            }
            ps->f16_bufs = new_bufs;
            ps->f16_bufs[ps->n_f16_bufs++] = data;
        }
        /* debug: if (i < 15) printf("[tensor] %s dims=[%lu,%lu]\n", n, (unsigned long)tinfo[i].dims[0], (unsigned long)tinfo[i].dims[1]); */
        int wired_before = wired;
        if (strcmp(n, "token_embd.weight") == 0) {
            ps->host_tok_emb = data;
            if (ps->host_vocab == 0) ps->host_vocab = (int)tinfo[i].dims[1];
            wired++;
        }
        else if (strcmp(n, "output_norm.weight") == 0) { ps->host_norm = data; wired++; }
        else if (strcmp(n, "output.weight") == 0) { ps->host_output = data; ps->host_output_dt = data_dt; wired++; }
        else {
            int l = layer_idx;
            if (is_layer_tensor) {
                if (strstr(n, "attn_q.weight")) { ps->host_layers[l].wq = data; ps->host_layers[l].wq_dt = data_dt; wired++; }
                else if (strstr(n, "attn_k.weight")) { ps->host_layers[l].wk = data; ps->host_layers[l].wk_dt = data_dt; wired++; }
                else if (strstr(n, "attn_v.weight")) { ps->host_layers[l].wv = data; ps->host_layers[l].wv_dt = data_dt; wired++; }
                else if (strstr(n, "attn_output.weight")) { ps->host_layers[l].wo = data; ps->host_layers[l].wo_dt = data_dt; wired++; }
                else if (strstr(n, "attn_q.bias")) { ps->host_layers[l].bq = data; wired++; }
                else if (strstr(n, "attn_k.bias")) { ps->host_layers[l].bk = data; wired++; }
                else if (strstr(n, "attn_v.bias")) { ps->host_layers[l].bv = data; wired++; }
                else if (strstr(n, "ffn_gate.weight") && !strstr(n, "ffn_gate_inp") && !strstr(n, "ffn_gate_up")) { ps->host_layers[l].ffn_gate = data; ps->host_layers[l].ffn_gate_dt = data_dt; wired++; }
                else if (strstr(n, "ffn_up.weight") && !strstr(n, "gate_up")) {
                    /* Check if fused gate+up: dims[1] > host_hidden means [dim, hidden*2] */
                    if (ps->host_hidden > 0 && (int)tinfo[i].dims[1] > ps->host_hidden * 3 / 2) {
                        ps->host_layers[l].ffn_gate_up = data; ps->host_layers[l].ffn_gate_up_dt = data_dt;
                    } else {
                        ps->host_layers[l].ffn_up = data; ps->host_layers[l].ffn_up_dt = data_dt;
                    }
                    wired++;
                }
                else if (strstr(n, "ffn_down.weight")) { ps->host_layers[l].ffn_down = data; ps->host_layers[l].ffn_down_dt = data_dt; wired++; }
                else if (strstr(n, "ffn_gate_up_proj") || strstr(n, "ffn_gate_up.weight")) { ps->host_layers[l].ffn_gate_up = data; ps->host_layers[l].ffn_gate_up_dt = data_dt; wired++; }
                else if (strstr(n, "attn_norm.weight")) { ps->host_layers[l].attn_norm = data; wired++; }
                else if (strstr(n, "ffn_norm.weight")) { ps->host_layers[l].ffn_norm = data; wired++; }
                else if (l == 0 && strstr(n, "ffn")) { printf("[doe] unwired FFN tensor: %s\n", n); }
            }
        }
        if (wired > wired_before) ps->host_n_params += n_elems;
    }
    if (getenv("DOE_PROFILE")) printf("[doe] wired %d host tensors\n", wired);
    free(tinfo);
    tinfo = NULL;

    /* tied embeddings: if output.weight missing, reuse token_embd.weight */
    if (!ps->host_output && ps->host_tok_emb) {
        ps->host_output = ps->host_tok_emb;
        printf("[doe] output.weight missing — using tied embeddings\n");
    }
    if (!ps->host_tok_emb || !ps->host_output || !ps->host_norm) {
        printf("[doe] host missing essential weights (tok_emb=%d out=%d norm=%d). abandoning.\n",
               ps->host_tok_emb!=NULL, ps->host_output!=NULL, ps->host_norm!=NULL);
        goto bail;
    }
    if (!doe_validate_host_config(ps, 1)) goto bail;

    if (!doe_validate_host_tensor_completeness(ps)) goto bail;

    /* ── Weight profiling — the sonar ── */
    printf("[sonar] profiling host weights...\n");
    ps->profile.n_layers = ps->host_n_layers;
    for (int l = 0; l < ps->host_n_layers && l < MAX_LAYERS; l++) {
        if (ps->host_layers[l].ffn_gate) {
            int gdt = ps->host_layers[l].ffn_gate_dt;
            if (gdt) {
                /* weight kept packed — dequant a temp f32 copy just for profiling */
                size_t ne = 0;
                float *tmp = NULL;
                if (profile_weight_shape_checked((size_t)ps->host_hidden, (size_t)ps->host_dim, &ne) &&
                    ne <= SIZE_MAX / sizeof(float)) {
                    tmp = malloc(ne * sizeof(float));
                }
                if (tmp) {
                    doe_dequant_to_f32((const uint8_t*)ps->host_layers[l].ffn_gate, gdt, (uint64_t)ne, tmp);
                    profile_weights(tmp, (size_t)ps->host_hidden, (size_t)ps->host_dim, &ps->profile.layers[l]);
                    free(tmp);
                } else memset(&ps->profile.layers[l], 0, sizeof(LayerProfile));
            } else {
                profile_weights(ps->host_layers[l].ffn_gate, (size_t)ps->host_hidden, (size_t)ps->host_dim, &ps->profile.layers[l]);
            }
        } else {
            memset(&ps->profile.layers[l], 0, sizeof(LayerProfile));
        }
    }
    float total_h = 0;
    for (int l = 0; l < ps->profile.n_layers; l++) total_h += ps->profile.layers[l].health;
    ps->profile.overall_health = total_h / (ps->profile.n_layers > 0 ? ps->profile.n_layers : 1);
    ps->profile.complexity = (float)ps->host_dim * ps->host_n_layers * ps->host_heads;
    ps->profile.fingerprint = compute_fingerprint(&ps->profile);

    printf("[sonar] host fingerprint: %016llx health=%.2f complexity=%.0f\n",
           (unsigned long long)ps->profile.fingerprint, ps->profile.overall_health, ps->profile.complexity);
    for (int l = 0; l < ps->host_n_layers && l < MAX_LAYERS; l++) {
        LayerProfile *lp = &ps->profile.layers[l];
        if (lp->l2_norm > 0)
            printf("  L%d: health=%.2f l2=%.2f std=%.4f sparse=%.1f%% dead=%d\n",
                   l, lp->health, lp->l2_norm, lp->std_dev, lp->sparsity*100, lp->dead_neurons);
    }

    /* ── Initialize living LoRA experts per layer ── */
    int initial_experts = ps->host_n_layers <= 8 ? 4 : ps->host_n_layers <= 16 ? 6 : 8;
    ps->n_field_layers = ps->host_n_layers;
    if (ps->n_field_layers > MAX_LAYERS) ps->n_field_layers = MAX_LAYERS;

    for (int l = 0; l < ps->n_field_layers; l++) {
        FieldLayer *fl = &ps->field_layers[l];
        fl->host_layer_idx = l;
        fl->n_alive = initial_experts;
        size_t vote_count = 0;
        if (ps->host_dim <= 0 ||
            !doe_mul_size((size_t)MAX_EXPERTS, (size_t)ps->host_dim, &vote_count) ||
            vote_count > (size_t)INT_MAX) {
            fprintf(stderr, "[doe] invalid parliament vote shape for layer %d (dim=%d)\n", l, ps->host_dim);
            goto bail;
        }
        fl->parliament.w_vote = calloc(vote_count, sizeof(float));
        if (!fl->parliament.w_vote) {
            fprintf(stderr, "[doe] parliament vote allocation failed for layer %d\n", l);
            goto bail;
        }
        float vote_std = 0.01f;
        for (size_t i = 0; i < vote_count; i++)
            fl->parliament.w_vote[i] = rand_normal() * vote_std;
        fl->parliament.consensus = 0.5f;
        /* Initialize experts with harmonic spacing — health-aware */
        float layer_health = ps->profile.layers[l].health;
        for (int e = 0; e < MAX_EXPERTS; e++) {
            if (e < initial_experts) {
                float freq = 6.2831853f * e / initial_experts;
                if (!init_lora_expert(&fl->experts[e], ps->host_dim, ps->lora_rank, freq))
                    goto bail;
                /* Weaker layers get stronger initial LoRA — DOE compensates */
                if (layer_health < 0.5f) {
                    float boost = (0.5f - layer_health) * 2.0f;
                    for (int i = 0; i < ps->host_dim * ps->lora_rank; i++) {
                        fl->experts[e].lora_A[i] *= (1.0f + boost);
                        fl->experts[e].lora_B[i] *= (1.0f + boost);
                    }
                }
            } else {
                memset(&fl->experts[e], 0, sizeof(LoraExpert));
            }
        }
    }

    /* NB: the parliament-on-GPU arena is packed AFTER mycelium_load (in main),
     * not here — a spore mutates w_vote/A/B, so packing must see the final
     * expert state. See field_repack_arena call after mycelium_load. */

    ps->active = 1;
    /* Build token hash table for O(1) lookup, then GPT-2 BPE scores */
    tok_ht_build(ps);
    build_gpt2_scores(ps);
    printf("[doe] attached to %s (arch=%s dim=%d layers=%d heads=%d kv=%d hd=%d vocab=%d %.1fMB)\n",
           path, ps->host_arch, ps->host_dim, ps->host_n_layers, ps->host_heads,
           ps->host_kv_heads, ps->host_head_dim, ps->host_vocab, (float)ps->mmap_size/(1024*1024));
    printf("[doe] rope_theta=%.0f rms_eps=%.1e bias=%s\n",
           ps->rope_theta, ps->rms_norm_eps,
           ps->host_layers[0].bq ? "yes" : "no");
    if (ps->is_gpt2_bpe) printf("[doe] tokenizer: GPT-2 BPE%s (%d merges)\n", ps->is_tekken ? "/Tekken" : "", ps->n_bpe_merges);
    /* Auto-detect nanollama chat style from identity tag or vocab tokens */
    if (ps->chat_style == 0 && ps->is_tekken && tok_lookup(ps, "[INST]", 6) >= 0)
        ps->chat_style = 2;
    if (ps->chat_style == 0 && (ps->identity_tag[0] ||
        tok_lookup(ps, "<|user_start|>", 14) >= 0)) ps->chat_style = 6;
    { const char *cs[] = {"raw","chatml","inst","zephyr","phi","gemma","nanollama"};
      printf("[doe] chat: %s\n", cs[ps->chat_style < 7 ? ps->chat_style : 0]); }
    printf("[doe] LoRA rank=%d alpha=%.2f experts=%d/layer — parliament is alive.\n",
           ps->lora_rank, ps->lora_alpha, initial_experts);
    #undef PC
    return 1;
bail:
    free(tinfo);
    tinfo = NULL;
    for (int l = 0; l < ps->n_field_layers; l++) {
        free(ps->field_layers[l].parliament.w_vote);
        ps->field_layers[l].parliament.w_vote = NULL;
        for (int e = 0; e < MAX_EXPERTS; e++)
            if (ps->field_layers[l].experts[e].alive)
                free_lora_expert(&ps->field_layers[l].experts[e]);
    }
    for (int i = 0; i < ps->n_f16_bufs; i++) free(ps->f16_bufs[i]);
    free(ps->f16_bufs); ps->f16_bufs = NULL; ps->n_f16_bufs = 0;
    doe_free_tokenizer_metadata(ps);
    if (ps->mmap_base) { munmap(ps->mmap_base, ps->mmap_size); ps->mmap_base = NULL; }
    printf("[doe] GGUF parse failed.\n");
    return 0;
}

static void index_free(GGUFIndex *ps) {
    for (int l = 0; l < ps->n_field_layers; l++) {
        free(ps->field_layers[l].parliament.w_vote);
        for (int e = 0; e < MAX_EXPERTS; e++)
            if (ps->field_layers[l].experts[e].alive)
                free_lora_expert(&ps->field_layers[l].experts[e]);
    }
    for (int i = 0; i < ps->n_f16_bufs; i++) free(ps->f16_bufs[i]);
    free(ps->f16_bufs);
    doe_free_tokenizer_metadata(ps);
    free(ps->gamma_data);
    ps->gamma_data = NULL;
    ps->gamma_size = 0;
    if (ps->mmap_base) munmap(ps->mmap_base, ps->mmap_size);
#ifdef USE_METAL
    /* expert_arena is wrapped by a NoCopy MTLBuffer (deallocator:nil) in notorch's
     * segment table and there is no per-region unregister. index_free runs once at
     * process exit (single-model lifecycle, :4171/:4221), so we deliberately do not
     * free the arena here — the OS reclaims it at exit, and the registered buffer is
     * guaranteed to outlive any GPU use. Freeing under the live NoCopy view would
     * leave a dangling segment. */
#endif
    memset(ps, 0, sizeof(GGUFIndex));
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PARLIAMENT ELECTION — variable-k over LoRA experts
 * ═══════════════════════════════════════════════════════════════════════════════ */
static int parliament_elect(Parliament *p, LoraExpert *experts, float *input, int dim,
                            HarmonicState *hs, int *selected, float *weights) {
    int n_alive = 0, alive_idx[MAX_EXPERTS];
    for (int e = 0; e < MAX_EXPERTS; e++) if (experts[e].alive) alive_idx[n_alive++] = e;
    if (n_alive < MIN_EXPERTS) return 0;

    float votes[MAX_EXPERTS]; float max_vote = -1e30f;
    for (int i = 0; i < n_alive; i++) {
        int e = alive_idx[i];
        float *row = p->w_vote + e * dim;
        float dot = 0;
        for (int j = 0; j < dim; j++) dot += row[j] * input[j];
        float res = expert_resonance(experts[e].frequency, hs);
        votes[e] = dot + 0.1f * res;
        if (!isfinite(votes[e])) votes[e] = -1e30f;
        if (votes[e] > max_vote) max_vote = votes[e];
    }
    float mean_v = 0;
    for (int i = 0; i < n_alive; i++) mean_v += votes[alive_idx[i]];
    mean_v /= n_alive;
    float var_v = 0;
    for (int i = 0; i < n_alive; i++) { float d = votes[alive_idx[i]] - mean_v; var_v += d*d; }
    var_v /= n_alive;
    float consensus = fminf(1.0f, sqrtf(var_v + 1e-8f) / (fabsf(mean_v) + 1.0f));
    if (!isfinite(consensus)) consensus = 0.5f;
    if (!isfinite(p->consensus) || p->consensus < 0.0f || p->consensus > 1.0f)
        p->consensus = 0.5f;
    p->consensus = 0.9f * p->consensus + 0.1f * consensus;
    if (!isfinite(p->consensus)) p->consensus = 0.5f;
    else if (p->consensus < 0.0f) p->consensus = 0.0f;
    else if (p->consensus > 1.0f) p->consensus = 1.0f;

    int k = (int)(n_alive * (1.0f - p->consensus));
    if (k < 2) k = 2; if (k > n_alive) k = n_alive;

    int used[MAX_EXPERTS] = {0};
    for (int ki = 0; ki < k; ki++) {
        float bv = -1e30f; int bi = 0;
        for (int i = 0; i < n_alive; i++) {
            int e = alive_idx[i];
            if (!used[e] && votes[e] > bv) { bv = votes[e]; bi = e; }
        }
        selected[ki] = bi; weights[ki] = votes[bi]; used[bi] = 1;
    }
    float mx = weights[0];
    for (int i = 1; i < k; i++) if (weights[i] > mx) mx = weights[i];
    float sum = 0;
    for (int i = 0; i < k; i++) { weights[i] = expf(weights[i]-mx); sum += weights[i]; }
    for (int i = 0; i < k; i++) weights[i] /= sum;
    p->election_count++;
    return k;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * NOTORCH — Hebbian plasticity for LoRA experts. from AML core.
 * no backprop. synapse strengthens from co-activation.
 * signal-gated: prophecy debt drives learning direction.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static int notorch_offset = 0; /* rotating window into LoRA rank */

static void notorch_step(float *A, float *B, int out_dim, int in_dim, int rank,
                         const float *x, const float *dy, float signal) {
    if (fabsf(signal) < 1e-8f) return;
    /* D-M2/Mythos: learn_signal (:3084) is asymmetric — punishment scales as -pd
     * (unbounded), reward is capped at 0.1. Clamp it so one high-debt step can't
     * overwrite the channel ~10x. Canon parity (am_notorch_step :7270). */
    if (signal >  2.0f) signal =  2.0f;
    if (signal < -2.0f) signal = -2.0f;
    float lr = F.notorch_lr * signal;
    /* NOTORCH operates at rank 4 but rotates across all LORA_RANK components.
     * each call updates 4 components starting at notorch_offset.
     * after rank/4 calls, every component has been updated once. */
    int nr = NOTORCH_RANK;
    if (nr > rank) nr = rank;
    int base = notorch_offset % rank;
    float u[NOTORCH_RANK];
    for (int j = 0; j < nr; j++) {
        int r = (base + j) % rank;
        float s = 0;
        for (int i = 0; i < out_dim && i < in_dim; i++) s += B[r * in_dim + i] * dy[i];
        u[j] = s + rand_normal() * 0.01f;
    }
    /* Oja's rule on both A and B: grow toward the co-activation (x, dy) through
     * the learned channel u, with the self-projection subtracted so each rank row
     * self-normalizes. The channel neither fades (the old bare *= decay) nor blows
     * up (canon's bare grow). This is what makes the parliament learn in-session
     * instead of eroding from init noise. */
    for (int j = 0; j < nr; j++) {
        int r = (base + j) % rank;
        float ur = u[j];
        for (int i = 0; i < in_dim; i++) {                 /* A = [in_dim, rank] */
            float *a = &A[i * rank + r];
            *a += lr * ur * (x[i] - ur * (*a));
        }
        for (int i = 0; i < out_dim && i < in_dim; i++) {  /* B = [rank, in_dim], apply layout */
            float *b = &B[r * in_dim + i];
            *b += lr * ur * (dy[i] - ur * (*b));
        }
    }
    notorch_offset = (base + nr) % rank;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * VITALITY + MITOSIS + APOPTOSIS — LoRA experts live and die
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void update_expert_vitality(FieldLayer *fl, int total_tokens) {
    int na = 0;
    for (int e = 0; e < MAX_EXPERTS; e++) if (fl->experts[e].alive) na++;
    if (na == 0) return;
    float fair = (float)total_tokens / na;
    for (int e = 0; e < MAX_EXPERTS; e++) {
        if (!fl->experts[e].alive) continue;
        LoraExpert *exp = &fl->experts[e];
        float ratio = fair > 0 ? (float)exp->tokens_seen / fair : 1.0f;
        exp->vitality += (ratio - 1.0f) * 0.05f;
        if (exp->vitality < 0) exp->vitality = 0;
        if (exp->vitality > 1) exp->vitality = 1;
        exp->age++;
        if (exp->vitality < 0.1f) exp->low_vitality_count++;
        else exp->low_vitality_count = 0;
        exp->tokens_seen = 0;
    }
    fl->n_alive = na;
}

static int try_mitosis(FieldLayer *fl, int dim, int rank) {
    int na = 0;
    for (int e = 0; e < MAX_EXPERTS; e++) if (fl->experts[e].alive) na++;
    if (na >= MAX_EXPERTS) return 0;
    int parent = -1;
    for (int e = 0; e < MAX_EXPERTS; e++) {
        if (!fl->experts[e].alive) continue;
        if (fl->experts[e].vitality > 0.8f && fl->experts[e].age > 20) { parent = e; break; }
    }
    if (parent < 0) return 0;
    int child = -1;
    for (int e = 0; e < MAX_EXPERTS; e++) if (!fl->experts[e].alive) { child = e; break; }
    if (child < 0) return 0;
    LoraExpert *p = &fl->experts[parent];
    float cf = p->frequency + 3.14159f / (na + 1);
    if (cf > 6.2831853f) cf -= 6.2831853f;
    if (!init_lora_expert(&fl->experts[child], dim, rank, cf)) return 0;
    LoraExpert *ch = &fl->experts[child];
    for (int i = 0; i < dim*rank; i++) ch->lora_A[i] = p->lora_A[i] + rand_normal()*0.01f;
    for (int i = 0; i < rank*dim; i++) ch->lora_B[i] = p->lora_B[i] + rand_normal()*0.01f;
    ch->vitality = 0.5f; p->vitality *= 0.8f;
    fl->n_alive++;
    return 1;
}

static int try_apoptosis(FieldLayer *fl) {
    int na = 0;
    for (int e = 0; e < MAX_EXPERTS; e++) if (fl->experts[e].alive) na++;
    if (na <= MIN_EXPERTS) return 0;
    for (int e = 0; e < MAX_EXPERTS; e++) {
        if (!fl->experts[e].alive) continue;
        if (fl->experts[e].low_vitality_count >= 8) {
            free_lora_expert(&fl->experts[e]);
            fl->n_alive--;
            return 1;
        }
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * CALENDAR DRIFT — 12D temporal self-awareness. from DOE m.c.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    float state[12]; int step;
} DriftSnapshot;

typedef struct {
    DriftSnapshot history[DRIFT_SNAPSHOTS];
    int head, n_snapshots;
    float drift, stability, drift_accel;
} CalendarDrift;

static void drift_init(CalendarDrift *cd) { memset(cd, 0, sizeof(CalendarDrift)); }

static void drift_snapshot(CalendarDrift *cd, float loss, GGUFIndex *ps, HarmonicState *hs) {
    DriftSnapshot *ds = &cd->history[cd->head % DRIFT_SNAPSHOTS];
    ds->step = F.step;
    int total_exp = 0;
    for (int l = 0; l < ps->n_field_layers; l++) total_exp += ps->field_layers[l].n_alive;
    ds->state[0] = (float)total_exp;
    ds->state[1] = ps->field_layers[0].parliament.consensus;
    ds->state[2] = loss;
    ds->state[3] = F.entropy;
    ds->state[4] = F.resonance;
    ds->state[5] = F.debt;
    ds->state[6] = hs->confidence;
    ds->state[7] = F.effective_temp;
    ds->state[8] = F.field_health;
    ds->state[9] = F.spring_energy;
    ds->state[10] = F.summer_energy;
    ds->state[11] = F.schumann_coherence;

    if (cd->n_snapshots > 0) {
        int prev = (cd->head - 1 + DRIFT_SNAPSHOTS) % DRIFT_SNAPSHOTS;
        float d2 = 0;
        for (int i = 0; i < 12; i++) {
            float diff = ds->state[i] - cd->history[prev].state[i];
            float range = fabsf(ds->state[i]) + 1e-8f;
            d2 += (diff / range) * (diff / range);
        }
        float new_drift = sqrtf(d2 / 12.0f);
        float prev_drift = cd->drift;
        cd->drift = 0.8f * cd->drift + 0.2f * new_drift;
        cd->drift_accel = cd->drift - prev_drift;
        cd->stability = 1.0f / (1.0f + cd->drift * 10.0f);
    }
    cd->head = (cd->head + 1) % DRIFT_SNAPSHOTS;
    if (cd->n_snapshots < DRIFT_SNAPSHOTS) cd->n_snapshots++;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * META-LEARNING — DOE learns from its own choices.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    int step; int n_experts; float consensus, loss, field_health;
    float prophecy_debt_avg; float drift; float delta_loss;
} MetaEntry;

typedef struct {
    MetaEntry history[META_HIST_CAP];
    int n_entries;
    float config_bias[4];
    float prediction_error;
} MetaTrack;

static void meta_init(MetaTrack *mt) {
    memset(mt, 0, sizeof(MetaTrack));
    for (int i = 0; i < 4; i++) mt->config_bias[i] = 0.5f;
}

static void meta_record(MetaTrack *mt, int step, int n_exp, float consensus,
                        float loss, float health, float debt_avg, float drift, float prev_loss) {
    if (mt->n_entries >= META_HIST_CAP) {
        memmove(mt->history, mt->history+1, (META_HIST_CAP-1)*sizeof(MetaEntry));
        mt->n_entries = META_HIST_CAP - 1;
    }
    MetaEntry *e = &mt->history[mt->n_entries];
    e->step = step; e->n_experts = n_exp; e->consensus = consensus;
    e->loss = loss; e->field_health = health; e->prophecy_debt_avg = debt_avg;
    e->drift = drift; e->delta_loss = prev_loss > 0 ? prev_loss - loss : 0;
    mt->n_entries++;
    if (mt->n_entries >= 2) {
        MetaEntry *prev = &mt->history[mt->n_entries-2];
        float improvement = prev->loss - loss;
        float lr_meta = 0.01f;
        float sig = improvement > 0 ? 1.0f : -0.5f;
        mt->config_bias[0] += lr_meta * sig * ((float)n_exp/MAX_EXPERTS - 0.5f);
        mt->config_bias[1] += lr_meta * sig * (consensus - 0.5f);
        mt->config_bias[2] += lr_meta * sig * (health - 0.5f);
        mt->config_bias[3] += lr_meta * sig * (debt_avg - 0.5f);
        for (int i = 0; i < 4; i++) {
            if (mt->config_bias[i] < 0.01f) mt->config_bias[i] = 0.01f;
            if (mt->config_bias[i] > 0.99f) mt->config_bias[i] = 0.99f;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MYCELIUM — LoRA spore forest.
 * DOE doesn't save full model GGUFs. it saves LoRA configurations:
 * the living experts, their weights, the parliament votes, the field state.
 * each spore is a snapshot of how DOE adapted to this host.
 * on restart with the same host (fingerprint match), load the best spore.
 * ═══════════════════════════════════════════════════════════════════════════════ */
#define MYCELIUM_DIR "doe_mycelium"

typedef struct {
    char path[256];
    uint64_t host_fingerprint;
    float fitness;
    int step;
} LoraSpore;

typedef struct {
    LoraSpore spores[MYCELIUM_MAX];
    int n_spores, best_idx;
} MyceliumState;

static void mycelium_init(MyceliumState *ms) {
    memset(ms, 0, sizeof(MyceliumState));
    ms->best_idx = -1;
    mkdir(MYCELIUM_DIR, 0755);
}

static void mycelium_save(GGUFIndex *ps, int step, float fitness) {
    /* A13: quarantine gate — never persist poisoned or dead parliament state. */
    if (!isfinite(fitness)) {
        printf("[mycelium] invalid spore fitness (%.6g) — spore NOT saved\n", fitness);
        return;
    }
    for (int l = 0; l < ps->n_field_layers; l++) {
        int actual_alive = field_recount_alive(&ps->field_layers[l]);
        if (actual_alive < MIN_EXPERTS) {
            printf("[mycelium] layer %d has only %d alive experts — spore NOT saved\n", l, actual_alive);
            return;
        }
        for (int e = 0; e < MAX_EXPERTS; e++)
            if (ps->field_layers[l].experts[e].alive &&
                lora_poisoned(ps->field_layers[l].experts[e].lora_A, ps->field_layers[l].experts[e].lora_B,
                              ps->host_dim * ps->lora_rank)) {
                printf("[mycelium] poisoned expert L%d e%d (NaN/Inf/|w|>1e4) — spore NOT saved (quarantine)\n", l, e);
                return;
            }
        float consensus = ps->field_layers[l].parliament.consensus;
        if (!isfinite(consensus) || consensus < 0.0f || consensus > 1.0f) {
            printf("[mycelium] invalid consensus in layer %d (%.6g) — spore NOT saved\n", l, consensus);
            return;
        }
    }
    char path[256];
    int pn = snprintf(path, sizeof(path), "%s/spore_%016llx_s%d.bin", MYCELIUM_DIR,
                      (unsigned long long)ps->profile.fingerprint, step);
    if (pn < 0 || pn >= (int)sizeof(path)) {
        printf("[mycelium] spore path too long\n");
        return;
    }
    char tmp_path[320];
    int tn = snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);
    if (tn < 0 || tn >= (int)sizeof(tmp_path)) {
        printf("[mycelium] temp spore path too long: %s\n", path);
        return;
    }
    int nl = ps->n_field_layers, dim = ps->host_dim, rank = ps->lora_rank;
    size_t vote_count = 0, lora_count = 0;
    if (dim <= 0 || rank <= 0 ||
        !doe_mul_size((size_t)MAX_EXPERTS, (size_t)dim, &vote_count) ||
        !doe_mul_size((size_t)dim, (size_t)rank, &lora_count) ||
        lora_count > (size_t)INT_MAX) {
        printf("[mycelium] invalid spore tensor shape (layers=%d dim=%d rank=%d)\n", nl, dim, rank);
        return;
    }
    errno = 0;
    FILE *f = fopen(tmp_path, "wb");
    if (!f) {
        int open_errno = errno;
        printf("[mycelium] cannot write %s%s%s\n", tmp_path,
               open_errno ? ": " : "", open_errno ? strerror(open_errno) : "");
        return;
    }
    #define SPORE_WRITE(ptr, size, count) do { \
        errno = 0; \
        if (fwrite((ptr), (size), (count), f) != (size_t)(count)) { \
            int saved_errno = errno; \
            printf("[mycelium] failed writing spore: %s%s%s\n", tmp_path, \
                   saved_errno ? ": " : "", saved_errno ? strerror(saved_errno) : ""); \
            fclose(f); \
            remove(tmp_path); \
            return; \
        } \
    } while (0)
    /* header: fingerprint, step, fitness, n_layers, dim, rank */
    uint64_t fp = ps->profile.fingerprint;
    SPORE_WRITE(&fp, 8, 1);
    SPORE_WRITE(&step, 4, 1);
    SPORE_WRITE(&fitness, 4, 1);
    SPORE_WRITE(&nl, 4, 1); SPORE_WRITE(&dim, 4, 1); SPORE_WRITE(&rank, 4, 1);
    /* per layer: n_alive, then per expert: alive, vitality, frequency, A, B */
    for (int l = 0; l < nl; l++) {
        FieldLayer *fl = &ps->field_layers[l];
        SPORE_WRITE(&fl->n_alive, 4, 1);
        /* parliament vote weights */
        SPORE_WRITE(fl->parliament.w_vote, sizeof(float), vote_count);
        SPORE_WRITE(&fl->parliament.consensus, 4, 1);
        for (int e = 0; e < MAX_EXPERTS; e++) {
            LoraExpert *ex = &fl->experts[e];
            SPORE_WRITE(&ex->alive, 4, 1);
            if (ex->alive) {
                SPORE_WRITE(&ex->vitality, 4, 1);
                SPORE_WRITE(&ex->frequency, 4, 1);
                SPORE_WRITE(ex->lora_A, sizeof(float), lora_count);
                SPORE_WRITE(ex->lora_B, sizeof(float), lora_count);
            }
        }
    }
    #undef SPORE_WRITE
    errno = 0;
    if (fclose(f) != 0) {
        int close_errno = errno;
        printf("[mycelium] failed closing spore: %s%s%s\n", tmp_path,
               close_errno ? ": " : "", close_errno ? strerror(close_errno) : "");
        remove(tmp_path);
        return;
    }
    errno = 0;
    if (rename(tmp_path, path) != 0) {
        int rename_errno = errno;
        printf("[mycelium] failed publishing spore %s -> %s%s%s\n", tmp_path, path,
               rename_errno ? ": " : "", rename_errno ? strerror(rename_errno) : "");
        remove(tmp_path);
        return;
    }
    printf("[mycelium] spore saved: %s (fitness=%.3f)\n", path, fitness);
}

static int mycelium_spore_step(const char *name, uint64_t target_fp, int *out_step) {
    if (!name || !out_step) return 0;
    char prefix[64];
    int plen_i = snprintf(prefix, sizeof(prefix), "spore_%016llx_s", (unsigned long long)target_fp);
    if (plen_i < 0 || plen_i >= (int)sizeof(prefix)) return 0;
    size_t plen = (size_t)plen_i;
    if (strncmp(name, prefix, plen) != 0) return 0;
    const char *step_s = name + plen;
    if (*step_s == '\0') return 0;
    if (*step_s < '0' || *step_s > '9') return 0;
    errno = 0;
    char *end = NULL;
    long step = strtol(step_s, &end, 10);
    if (end == step_s || errno != 0 || step < 0 || step > INT_MAX) return 0;
    if (strcmp(end, ".bin") != 0) return 0;
    *out_step = (int)step;
    return 1;
}

static int lora_spore_step_desc_cmp(const void *a, const void *b) {
    const LoraSpore *sa = (const LoraSpore *)a;
    const LoraSpore *sb = (const LoraSpore *)b;
    return (sb->step > sa->step) - (sb->step < sa->step);
}

static int mycelium_load_file(GGUFIndex *ps, uint64_t target_fp, const char *spore_path, int filename_step) {
    errno = 0;
    FILE *f = fopen(spore_path, "rb");
    if (!f) {
        int open_errno = errno;
        printf("[mycelium] cannot read spore %s%s%s\n", spore_path,
               open_errno ? ": " : "", open_errno ? strerror(open_errno) : "");
        return 0;
    }
    FieldLayer tmp_layers[MAX_LAYERS];
    memset(tmp_layers, 0, sizeof(tmp_layers));
    int loaded = 0;
    #define SPORE_READ(ptr, size, count) do { \
        if (fread((ptr), (size), (count), f) != (size_t)(count)) { \
            printf("[mycelium] truncated spore: %s\n", spore_path); \
            goto done; \
        } \
    } while (0)
    uint64_t fp; SPORE_READ(&fp, 8, 1);
    if (fp != target_fp) goto done;
    int step; float fitness;
    SPORE_READ(&step, 4, 1); SPORE_READ(&fitness, 4, 1);
    if (step != filename_step) {
        printf("[mycelium] spore step mismatch (%d filename, %d header): %s\n",
               filename_step, step, spore_path);
        goto done;
    }
    if (!isfinite(fitness)) {
        printf("[mycelium] invalid spore fitness in %s; refusing load\n", spore_path);
        goto done;
    }
    int nl, dim, rank;
    SPORE_READ(&nl, 4, 1); SPORE_READ(&dim, 4, 1); SPORE_READ(&rank, 4, 1);
    if (nl != ps->n_field_layers || dim != ps->host_dim || rank != ps->lora_rank) {
        printf("[mycelium] spore mismatch (layers=%d/%d dim=%d/%d rank=%d/%d)\n",
               nl, ps->n_field_layers, dim, ps->host_dim, rank, ps->lora_rank);
        goto done;
    }
    size_t vote_count = 0, vote_bytes = 0, lora_count = 0;
    if (dim <= 0 || rank <= 0 ||
        !doe_mul_size((size_t)MAX_EXPERTS, (size_t)dim, &vote_count) ||
        !doe_mul_size(vote_count, sizeof(float), &vote_bytes) ||
        !doe_mul_size((size_t)dim, (size_t)rank, &lora_count) ||
        lora_count > (size_t)INT_MAX) {
        printf("[mycelium] invalid spore tensor shape in %s (layers=%d dim=%d rank=%d)\n",
               spore_path, nl, dim, rank);
        goto done;
    }
    for (int l = 0; l < nl; l++) {
        FieldLayer *fl = &tmp_layers[l];
        fl->host_layer_idx = ps->field_layers[l].host_layer_idx;
        fl->parliament.w_vote = malloc(vote_bytes);
        if (!fl->parliament.w_vote) {
            printf("[mycelium] cannot allocate spore vote rows for layer %d\n", l);
            goto done;
        }
        int saved_alive = 0;
        SPORE_READ(&saved_alive, 4, 1);
        SPORE_READ(fl->parliament.w_vote, sizeof(float), vote_count);
        SPORE_READ(&fl->parliament.consensus, 4, 1);
        if (!isfinite(fl->parliament.consensus) ||
            fl->parliament.consensus < 0.0f || fl->parliament.consensus > 1.0f) {
            printf("[mycelium] invalid consensus in spore layer %d; resetting to 0.5\n", l);
            fl->parliament.consensus = 0.5f;
        }
        for (int e = 0; e < MAX_EXPERTS; e++) {
            LoraExpert *ex = &fl->experts[e];
            int alive; SPORE_READ(&alive, 4, 1);
            if (alive != 0 && alive != 1) {
                printf("[mycelium] non-canonical alive flag in spore layer %d expert %d (%d); refusing load\n",
                       l, e, alive);
                goto done;
            }
            if (alive) {
                float vitality, frequency;
                SPORE_READ(&vitality, 4, 1);
                SPORE_READ(&frequency, 4, 1);
                vitality = sanitize_expert_vitality(vitality, l, e);
                frequency = sanitize_expert_frequency(frequency, l, e);
                if (!ex->alive) {
                    if (!init_lora_expert(ex, dim, rank, frequency)) {
                        goto done;
                    }
                }
                ex->alive = 1;
                ex->vitality = vitality;
                ex->frequency = frequency;
                SPORE_READ(ex->lora_A, sizeof(float), lora_count);
                SPORE_READ(ex->lora_B, sizeof(float), lora_count);
                if (lora_poisoned(ex->lora_A, ex->lora_B, (int)lora_count)) {
                    printf("[mycelium] poisoned expert payload in spore L%d e%d; dropping expert\n", l, e);
                    free_lora_expert(ex);
                }
            } else if (ex->alive) {
                free_lora_expert(ex);
            }
        }
        int actual_alive = field_recount_alive(fl);
        if (saved_alive != actual_alive) {
            printf("[mycelium] spore alive count mismatch layer %d (%d header, %d flags); using flags\n",
                   l, saved_alive, actual_alive);
        }
        if (actual_alive < MIN_EXPERTS) {
            printf("[mycelium] spore layer %d has only %d alive experts after validation; refusing load\n",
                   l, actual_alive);
            goto done;
        }
    }
    errno = 0;
    int extra = fgetc(f);
    if (extra != EOF) {
        printf("[mycelium] trailing data after spore payload: %s\n", spore_path);
        goto done;
    }
    if (ferror(f)) {
        int read_errno = errno;
        printf("[mycelium] read error at spore end %s%s%s\n", spore_path,
               read_errno ? ": " : "", read_errno ? strerror(read_errno) : "");
        goto done;
    }
    errno = 0;
    if (fclose(f) != 0) {
        int close_errno = errno;
        f = NULL;
        printf("[mycelium] failed closing spore after read: %s%s%s\n", spore_path,
               close_errno ? ": " : "", close_errno ? strerror(close_errno) : "");
        goto done;
    }
    f = NULL;
    for (int l = 0; l < nl; l++) {
        field_release_resources(&ps->field_layers[l]);
        ps->field_layers[l] = tmp_layers[l];
        memset(&tmp_layers[l], 0, sizeof(tmp_layers[l]));
    }
    printf("[mycelium] spore loaded: %s (step=%d fitness=%.3f)\n", spore_path, step, fitness);
    loaded = 1;
done:
    #undef SPORE_READ
    if (f) fclose(f);
    for (int l = 0; l < MAX_LAYERS; l++)
        field_release_resources(&tmp_layers[l]);
    return loaded;
}

static int mycelium_load(GGUFIndex *ps, uint64_t target_fp) {
    LoraSpore *candidates = NULL;
    int n_candidates = 0, cap_candidates = 0;
    DIR *dir = opendir(MYCELIUM_DIR);
    if (!dir) return 0;
    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        const char *name = ent->d_name;
        int s = -1;
        if (!mycelium_spore_step(name, target_fp, &s)) continue;
        char path[256];
        int npath = snprintf(path, sizeof(path), "%s/%s", MYCELIUM_DIR, name);
        if (npath < 0 || npath >= (int)sizeof(path)) {
            printf("[mycelium] spore path too long, skipping: %s\n", name);
            continue;
        }
        if (n_candidates == cap_candidates) {
            if (cap_candidates > INT_MAX / 2 ||
                (cap_candidates > 0 && (size_t)cap_candidates * 2 > SIZE_MAX / sizeof(*candidates))) {
                printf("[mycelium] too many spore candidates; refusing partial fallback list\n");
                closedir(dir);
                free(candidates);
                return 0;
            }
            int new_cap = cap_candidates ? cap_candidates * 2 : 16;
            errno = 0;
            LoraSpore *grown = realloc(candidates, (size_t)new_cap * sizeof(*candidates));
            if (!grown) {
                int saved_errno = errno;
                printf("[mycelium] cannot allocate spore candidate list (%d entries)%s%s\n",
                       new_cap, saved_errno ? ": " : "", saved_errno ? strerror(saved_errno) : "");
                closedir(dir);
                free(candidates);
                return 0;
            }
            candidates = grown;
            cap_candidates = new_cap;
        }
        LoraSpore *cand = &candidates[n_candidates++];
        memset(cand, 0, sizeof(*cand));
        snprintf(cand->path, sizeof(cand->path), "%s", path);
        cand->host_fingerprint = target_fp;
        cand->step = s;
    }
    closedir(dir);
    if (n_candidates == 0) {
        free(candidates);
        return 0;
    }

    qsort(candidates, (size_t)n_candidates, sizeof(*candidates), lora_spore_step_desc_cmp);

    for (int i = 0; i < n_candidates; i++) {
        if (mycelium_load_file(ps, target_fp, candidates[i].path, candidates[i].step)) {
            free(candidates);
            return 1;
        }
        if (i + 1 < n_candidates)
            printf("[mycelium] trying older spore after rejecting %s\n", candidates[i].path);
    }
    free(candidates);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * INDEX FORWARD — run token through host with DOE modulation.
 *
 * per layer:
 *   1. host attention (read-only weights, KV cache)
 *   2. parliament election (which LoRA experts vote)
 *   3. Delta Voice injection: x += Σ(w_k × α × A_k @ (B_k @ x))
 *   4. host FFN (read-only)
 *   5. layer_focus scaling on residual
 *
 * after all layers:
 *   6. field modulation on logits
 *   7. prophecy debt computation
 *   8. NOTORCH Hebbian update on winning experts
 *
 * the host swims. the field steers. nobody knows who's in charge.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    float *x, *xb, *xb2, *q, *k, *v, *att, *logits;
    float *hb, *hb2, *expert_out;
    float *key_cache, *value_cache;
    float *cos_cache, *sin_cache;
    HarmonicState hs;
    int max_seq;
    size_t kv_bytes;
    float *img_embeds;            /* native pixtral image embeds [host_dim * img_count], or NULL */
    int    img_start, img_count;  /* positions [img_start, img_start+img_count) splice img_embeds */
} InferState;

static void free_infer(InferState *s);

static InferState alloc_infer(GGUFIndex *ps, int max_seq) {
    InferState s = {0};
    int D = ps->host_dim, kd = ps->host_kv_heads * ps->host_head_dim;
    int H = ps->host_hidden;
    size_t kv_elems = 0, kv_bytes = 0;
    if (max_seq <= 0 ||
        !doe_mul_size((size_t)ps->host_n_layers, (size_t)max_seq, &kv_elems) ||
        !doe_mul_size(kv_elems, (size_t)kd, &kv_elems) ||
        !doe_mul_size(kv_elems, sizeof(float), &kv_bytes)) {
        fprintf(stderr, "[doe] invalid inference allocation size\n");
        return s;
    }
    s.max_seq = max_seq;
    s.kv_bytes = kv_bytes;
    s.x = calloc((size_t)D, sizeof(float)); s.xb = calloc((size_t)D, sizeof(float)); s.xb2 = calloc((size_t)D, sizeof(float));
    s.q = calloc((size_t)ps->host_heads * ps->host_head_dim, sizeof(float));
    s.k = calloc((size_t)kd, sizeof(float)); s.v = calloc((size_t)kd, sizeof(float));
    s.att = calloc((size_t)ps->host_heads * max_seq, sizeof(float));
    s.logits = calloc((size_t)ps->host_vocab, sizeof(float));
    s.hb = calloc((size_t)H, sizeof(float)); s.hb2 = calloc((size_t)H * 2, sizeof(float)); /* *2 for fused gate_up */
    s.expert_out = calloc((size_t)D, sizeof(float));
    if (!s.x || !s.xb || !s.xb2 || !s.q || !s.k || !s.v || !s.att ||
        !s.logits || !s.hb || !s.hb2 || !s.expert_out) {
        fprintf(stderr, "[doe] inference activation allocation failed\n");
        free_infer(&s);
        return s;
    }
    /* M4 stage (a): KV-cache on a page-aligned region so the GPU attn_decode
     * kernel can later resolve it by offset (nt_metal_register_region). Apple
     * Silicon page = 16384 (getpagesize()), not 4096. CPU attention still reads
     * it exactly as before — this stage only changes alloc + registration. */
    if (posix_memalign((void**)&s.key_cache,   (size_t)getpagesize(), kv_bytes) ||
        posix_memalign((void**)&s.value_cache, (size_t)getpagesize(), kv_bytes)) {
        fprintf(stderr, "[doe] KV posix_memalign failed — falling back to calloc\n");
        free(s.key_cache); s.key_cache = NULL;
        free(s.value_cache); s.value_cache = NULL;
        s.key_cache = calloc(kv_elems, sizeof(float));
        s.value_cache = calloc(kv_elems, sizeof(float));
    } else {
        memset(s.key_cache, 0, kv_bytes);
        memset(s.value_cache, 0, kv_bytes);
#ifdef USE_METAL
        if (nt_metal_available()) {
            nt_metal_register_region(s.key_cache, kv_bytes);
            nt_metal_register_region(s.value_cache, kv_bytes);
        }
#endif
    }
    int half = ps->host_head_dim / 2;
    if (!s.key_cache || !s.value_cache) {
        fprintf(stderr, "[doe] KV cache allocation failed\n");
        free_infer(&s);
        return s;
    }
    s.cos_cache = calloc((size_t)max_seq * half, sizeof(float));
    s.sin_cache = calloc((size_t)max_seq * half, sizeof(float));
    if (!s.cos_cache || !s.sin_cache) {
        fprintf(stderr, "[doe] RoPE cache allocation failed\n");
        free_infer(&s);
        return s;
    }
    float rope_theta = ps->rope_theta;
    for (int p = 0; p < max_seq; p++)
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(rope_theta, (float)(2*i) / (float)ps->host_head_dim);
            float ang = (float)p * freq;
            s.cos_cache[p*half+i] = cosf(ang);
            s.sin_cache[p*half+i] = sinf(ang);
        }
    return s;
}

static void free_infer(InferState *s) {
    free(s->img_embeds);
    free(s->x); free(s->xb); free(s->xb2);
    free(s->q); free(s->k); free(s->v);
    free(s->att); free(s->logits);
    free(s->hb); free(s->hb2); free(s->expert_out);
    free(s->key_cache); free(s->value_cache);
    free(s->cos_cache); free(s->sin_cache);
    memset(s, 0, sizeof(InferState));
}

static void infer_clear_image_embeds(InferState *s) {
    free(s->img_embeds);
    s->img_embeds = NULL;
    s->img_start = 0;
    s->img_count = 0;
}

#if defined(USE_METAL) || defined(DOE_TESTING)
static void doe_metal_slot_failure_diagnostic(const char *scope, const char *op, int rc) {
    fprintf(stderr, "[doe] Metal slot path failed in %s at %s (rc=%d); aborting batch and falling back to CPU\n",
            scope, op ? op : "(unknown)", rc);
}
#endif

#ifdef DOE_TESTING
static int g_test_forced_slot_fail_emitted = 0;
static void doe_test_maybe_force_slot_failure(void) {
    const char *op = getenv("DOE_TEST_FORCE_SLOT_FAIL");
    if (!op || !*op || g_test_forced_slot_fail_emitted) return;
    g_test_forced_slot_fail_emitted = 1;
    doe_metal_slot_failure_diagnostic("test-forced slot path", op, 77);
}
#endif

#ifdef USE_METAL
static void doe_metal_slot_fail(GGUFIndex *ps, int *disabled, const char *scope, const char *op, int rc) {
    doe_metal_slot_failure_diagnostic(scope, op, rc);
    nt_metal_abort_slots(); /* drops the active slot batch/table without tearing down registered Metal regions */
    if (ps) ps->cons_slot_init = 0;
    if (disabled) *disabled = 1;
}

#define DOE_METAL_TRY(label, call) do { \
    int _rc = (call); \
    if (_rc != 0) { metal_rc = _rc; metal_op = #call; goto label; } \
} while (0)
#endif

static float *doe_forward(GGUFIndex *ps, InferState *s, int token, int pos) {
    int D = ps->host_dim, hd = ps->host_head_dim;
    int kd = ps->host_kv_heads * hd;
    int H = ps->host_hidden;
    int hg = ps->host_heads / ps->host_kv_heads;
    float sc = 1.0f / sqrtf((float)hd);
#ifdef DOE_TESTING
    doe_test_maybe_force_slot_failure();
#endif
    /* per-shape dims for DOE_PERSHAPE matvec grouping (keys pershape_group). */
    g_ps_D = D; g_ps_qn = ps->host_heads*hd; g_ps_kd = kd; g_ps_H = H; g_ps_H2 = H*2; g_ps_vocab = ps->host_vocab;

    /* Embedding (image-aware: splice native pixtral embeds at image positions) */
    if (s->img_embeds && pos >= s->img_start && pos < s->img_start + s->img_count)
        memcpy(s->x, s->img_embeds + (size_t)(pos - s->img_start) * D, D * sizeof(float));
    else if (token >= 0 && token < ps->host_vocab)
        memcpy(s->x, ps->host_tok_emb + token * D, D * sizeof(float));
    else
        memset(s->x, 0, D * sizeof(float));

    /* ── M4 stage (г): resident decode — the whole token in ONE command buffer.
     * X lives in SLOT_X on the GPU across all layers; 1 upload + 1 commit + 1
     * download per token instead of 2 commits + 2 CPU round-trips per layer
     * (~80 -> ~1 GPU syncs). Requires lora_alpha==0 (no CPU parliament step
     * between attn and ffn) and every layer slot-eligible. The serial-dispatch
     * encoder serialises the reused slots, so it is bit-identical to the
     * per-layer slot path. DOE_NO_RESIDENT falls back. */
    int resident_done = 0, head_done = 0;
#ifdef USE_METAL
    int metal_slots_disabled = 0;
    /* alpha==0: pure fast path. alpha!=0 with a registered expert arena: the
     * parliament (election + LoRA inject) runs on the GPU between attn and ffn,
     * so x stays resident and the token is still one command buffer. */
    if (!metal_slots_disabled && !getenv("DOE_NO_SLOTS") && !getenv("DOE_NO_RESIDENT") && nt_metal_available() &&
        (ps->lora_alpha == 0.0f || (!g_train && ps->expert_arena_ready))) {
        int eligible = 1;
        for (int l = 0; l < ps->host_n_layers && l < MAX_LAYERS; l++) {
            if (!ps->host_layers[l].wq) continue;
            int qd=ps->host_layers[l].wq_dt, kdt=ps->host_layers[l].wk_dt,
                vd=ps->host_layers[l].wv_dt, od=ps->host_layers[l].wo_dt,
                ddt=ps->host_layers[l].ffn_down_dt;
            int fused = ps->host_layers[l].ffn_gate_up != NULL;
            int gdt = fused ? ps->host_layers[l].ffn_gate_up_dt : ps->host_layers[l].ffn_gate_dt;
            int udt = fused ? gdt : ps->host_layers[l].ffn_up_dt;
            int has_ffn = ps->host_layers[l].ffn_down &&
                          (fused || (ps->host_layers[l].ffn_gate && ps->host_layers[l].ffn_up));
            if (!ps->host_layers[l].attn_norm || !ps->host_layers[l].ffn_norm || !has_ffn ||
                ps->host_layers[l].bq || ps->host_layers[l].bk || ps->host_layers[l].bv ||
                !((qd==12||qd==14)&&(kdt==12||kdt==14)&&(vd==12||vd==14)&&(od==12||od==14)&&
                  (gdt==12||gdt==14)&&(udt==12||udt==14)&&(ddt==12||ddt==14))) { eligible = 0; break; }
        }
        if (eligible) {
            int metal_rc = 0;
            const char *metal_op = NULL;
            struct timespec _r0; clock_gettime(CLOCK_MONOTONIC, &_r0);
            int qn = ps->host_heads * hd;
            DOE_METAL_TRY(resident_fail, nt_metal_slot_alloc(SLOT_X, D*4));
            DOE_METAL_TRY(resident_fail, nt_metal_slot_alloc(SLOT_FN, D*4));
            DOE_METAL_TRY(resident_fail, nt_metal_slot_alloc(SLOT_Q, qn*4));
            DOE_METAL_TRY(resident_fail, nt_metal_slot_alloc(SLOT_K, kd*4));
            DOE_METAL_TRY(resident_fail, nt_metal_slot_alloc(SLOT_V, kd*4));
            DOE_METAL_TRY(resident_fail, nt_metal_slot_alloc(SLOT_O, qn*4));
            DOE_METAL_TRY(resident_fail, nt_metal_slot_alloc(SLOT_XB, D*4));
            DOE_METAL_TRY(resident_fail, nt_metal_slot_alloc(SLOT_G, H*4));
            DOE_METAL_TRY(resident_fail, nt_metal_slot_alloc(SLOT_U, H*4));
            DOE_METAL_TRY(resident_fail, nt_metal_slot_alloc(SLOT_HB, H*4));
            DOE_METAL_TRY(resident_fail, nt_metal_slot_alloc(SLOT_DN, D*4));
            DOE_METAL_TRY(resident_fail, nt_metal_slot_upload(SLOT_X, s->x, D*4));
            /* parliament-on-GPU setup (alpha != 0): per-token resonance + frozen
             * alive mask + persistent consensus, all resident so the election runs
             * in-batch. freq[e]=2*pi*e/initial_experts and alive[e] are layer-
             * independent at inference, so layer 0 is representative. */
            int parl = (ps->lora_alpha != 0.0f) && !g_train && ps->expert_arena_ready;
            if (parl) {
                int ne = MAX_EXPERTS;
                DOE_METAL_TRY(resident_fail, nt_metal_slot_alloc(SLOT_PTMP,  (uint64_t)ne*ps->lora_rank*4));
                DOE_METAL_TRY(resident_fail, nt_metal_slot_alloc(SLOT_PGATE, (uint64_t)ne*4));
                DOE_METAL_TRY(resident_fail, nt_metal_slot_alloc(SLOT_PVOTES,(uint64_t)ne*4));
                DOE_METAL_TRY(resident_fail, nt_metal_slot_alloc(SLOT_PRES,  (uint64_t)ne*4));
                DOE_METAL_TRY(resident_fail, nt_metal_slot_alloc(SLOT_PALIVE,(uint64_t)ne*4));
                DOE_METAL_TRY(resident_fail, nt_metal_slot_alloc(SLOT_PCONS, (uint64_t)MAX_LAYERS*4));
                float res[MAX_EXPERTS], alv[MAX_EXPERTS];
                FieldLayer *fl0 = &ps->field_layers[0];
                for (int e = 0; e < ne; e++) {
                    res[e] = 0.1f * expert_resonance(fl0->experts[e].frequency, &s->hs);
                    alv[e] = fl0->experts[e].alive ? 1.0f : 0.0f;
                }
                DOE_METAL_TRY(resident_fail, nt_metal_slot_upload(SLOT_PRES,   res, (uint64_t)ne*4));
                DOE_METAL_TRY(resident_fail, nt_metal_slot_upload(SLOT_PALIVE, alv, (uint64_t)ne*4));
                if (!ps->cons_slot_init) {
                    float cons[MAX_LAYERS];
                    for (int l = 0; l < MAX_LAYERS; l++)
                        cons[l] = (l < ps->n_field_layers) ? ps->field_layers[l].parliament.consensus : 0.5f;
                    DOE_METAL_TRY(resident_fail, nt_metal_slot_upload(SLOT_PCONS, cons, (uint64_t)MAX_LAYERS*4));
                    ps->cons_slot_init = 1;
                }
            }
            DOE_METAL_TRY(resident_fail, nt_metal_batch_begin());
            for (int l = 0; l < ps->host_n_layers && l < MAX_LAYERS; l++) {
                if (!ps->host_layers[l].wq) continue;
                int qd=ps->host_layers[l].wq_dt, kdt=ps->host_layers[l].wk_dt,
                    vd=ps->host_layers[l].wv_dt, od=ps->host_layers[l].wo_dt,
                    ddt=ps->host_layers[l].ffn_down_dt;
                int co2 = l * s->max_seq * kd + pos * kd;
                float *Kbase = s->key_cache   + (size_t)l * s->max_seq * kd;
                float *Vbase = s->value_cache + (size_t)l * s->max_seq * kd;
                const uint8_t *Wdn = (const uint8_t*)ps->host_layers[l].ffn_down;
                /* attention half-layer */
                DOE_METAL_TRY(resident_fail, nt_metal_rmsnorm(SLOT_FN, SLOT_X, ps->host_layers[l].attn_norm, D, ps->rms_norm_eps));
                if (qd==14)  { DOE_METAL_TRY(resident_fail, nt_metal_q6k_matvec_slot(SLOT_Q,(const uint8_t*)ps->host_layers[l].wq,SLOT_FN,qn,D)); }
                else         { DOE_METAL_TRY(resident_fail, nt_metal_q4k_matvec_slot(SLOT_Q,(const uint8_t*)ps->host_layers[l].wq,SLOT_FN,qn,D)); }
                if (kdt==14) { DOE_METAL_TRY(resident_fail, nt_metal_q6k_matvec_slot(SLOT_K,(const uint8_t*)ps->host_layers[l].wk,SLOT_FN,kd,D)); }
                else         { DOE_METAL_TRY(resident_fail, nt_metal_q4k_matvec_slot(SLOT_K,(const uint8_t*)ps->host_layers[l].wk,SLOT_FN,kd,D)); }
                if (vd==14)  { DOE_METAL_TRY(resident_fail, nt_metal_q6k_matvec_slot(SLOT_V,(const uint8_t*)ps->host_layers[l].wv,SLOT_FN,kd,D)); }
                else         { DOE_METAL_TRY(resident_fail, nt_metal_q4k_matvec_slot(SLOT_V,(const uint8_t*)ps->host_layers[l].wv,SLOT_FN,kd,D)); }
                int rope_norm = g_rope_norm || arch_rope_norm(ps->host_arch);
                DOE_METAL_TRY(resident_fail, nt_metal_rope(SLOT_Q, ps->host_heads,    hd, pos, ps->rope_theta, rope_norm));
                DOE_METAL_TRY(resident_fail, nt_metal_rope(SLOT_K, ps->host_kv_heads, hd, pos, ps->rope_theta, rope_norm));
                DOE_METAL_TRY(resident_fail, nt_metal_copy_to_region(s->key_cache   + co2, SLOT_K, kd*4));
                DOE_METAL_TRY(resident_fail, nt_metal_copy_to_region(s->value_cache + co2, SLOT_V, kd*4));
                DOE_METAL_TRY(resident_fail, nt_metal_attn_decode(SLOT_O, SLOT_Q, Kbase, Vbase, pos+1,
                                                                  ps->host_heads, ps->host_kv_heads, hd,
                                                                  (uint32_t)kd, (uint32_t)hd, (uint32_t)kd, (uint32_t)hd, sc));
                if (od==14)  { DOE_METAL_TRY(resident_fail, nt_metal_q6k_matvec_slot(SLOT_XB,(const uint8_t*)ps->host_layers[l].wo,SLOT_O,D,qn)); }
                else         { DOE_METAL_TRY(resident_fail, nt_metal_q4k_matvec_slot(SLOT_XB,(const uint8_t*)ps->host_layers[l].wo,SLOT_O,D,qn)); }
                DOE_METAL_TRY(resident_fail, nt_metal_add(SLOT_X, SLOT_X, SLOT_XB, D));
                /* parliament: election (w_vote@x -> variable-k gate) + LoRA inject,
                 * x += alpha * sum_k w_k * A_k @ (B_k @ x), all on the resident x. */
                if (parl) {
                    const float *lb = ps->expert_arena + (size_t)l * ps->expert_layer_floats;
                    DOE_METAL_TRY(resident_fail, nt_metal_parliament_votes(lb, SLOT_X, SLOT_PVOTES, D, MAX_EXPERTS));
                    DOE_METAL_TRY(resident_fail, nt_metal_parliament_elect(SLOT_PVOTES, SLOT_PRES, SLOT_PALIVE, SLOT_PCONS,
                                                                           SLOT_PGATE, MAX_EXPERTS, l, MIN_EXPERTS));
                    DOE_METAL_TRY(resident_fail, nt_metal_parliament_inject(SLOT_X, SLOT_PTMP, SLOT_PGATE, lb,
                                                                            D, ps->lora_rank, MAX_EXPERTS, ps->lora_alpha));
                }
                /* ffn half-layer (fused gate_up or separate gate + up) */
                DOE_METAL_TRY(resident_fail, nt_metal_rmsnorm(SLOT_FN, SLOT_X, ps->host_layers[l].ffn_norm, D, ps->rms_norm_eps));
                if (ps->host_layers[l].ffn_gate_up) {
                    int gdt = ps->host_layers[l].ffn_gate_up_dt;
                    size_t gu_row = quant_raw_bytes(gdt, D);
                    const uint8_t *Wgu = (const uint8_t*)ps->host_layers[l].ffn_gate_up;
                    if (gdt==14) { DOE_METAL_TRY(resident_fail, nt_metal_q6k_matvec_slot(SLOT_G, Wgu, SLOT_FN, H, D));
                                   DOE_METAL_TRY(resident_fail, nt_metal_q6k_matvec_slot(SLOT_U, Wgu+(size_t)H*gu_row, SLOT_FN, H, D)); }
                    else         { DOE_METAL_TRY(resident_fail, nt_metal_q4k_matvec_slot(SLOT_G, Wgu, SLOT_FN, H, D));
                                   DOE_METAL_TRY(resident_fail, nt_metal_q4k_matvec_slot(SLOT_U, Wgu+(size_t)H*gu_row, SLOT_FN, H, D)); }
                } else {
                    const uint8_t *Wg = (const uint8_t*)ps->host_layers[l].ffn_gate;
                    const uint8_t *Wu = (const uint8_t*)ps->host_layers[l].ffn_up;
                    int g2 = ps->host_layers[l].ffn_gate_dt, u2 = ps->host_layers[l].ffn_up_dt;
                    if (g2==14) { DOE_METAL_TRY(resident_fail, nt_metal_q6k_matvec_slot(SLOT_G, Wg, SLOT_FN, H, D)); }
                    else        { DOE_METAL_TRY(resident_fail, nt_metal_q4k_matvec_slot(SLOT_G, Wg, SLOT_FN, H, D)); }
                    if (u2==14) { DOE_METAL_TRY(resident_fail, nt_metal_q6k_matvec_slot(SLOT_U, Wu, SLOT_FN, H, D)); }
                    else        { DOE_METAL_TRY(resident_fail, nt_metal_q4k_matvec_slot(SLOT_U, Wu, SLOT_FN, H, D)); }
                }
                DOE_METAL_TRY(resident_fail, nt_metal_silu_mul(SLOT_HB, SLOT_G, SLOT_U, H));
                if (ddt==14) { DOE_METAL_TRY(resident_fail, nt_metal_q6k_matvec_slot(SLOT_DN, Wdn, SLOT_HB, D, H)); }
                else         { DOE_METAL_TRY(resident_fail, nt_metal_q4k_matvec_slot(SLOT_DN, Wdn, SLOT_HB, D, H)); }
                DOE_METAL_TRY(resident_fail, nt_metal_add(SLOT_X, SLOT_X, SLOT_DN, D));
            }
            /* fold final norm + lm_head into the same command buffer when the
             * output weight is slot-matvec-able: norm -> lm_head on the GPU,
             * download logits (vocab) instead of x (dim). One fewer solo
             * matvec + round-trip. CPU lm_head path runs only when not folded. */
            int hf = ps->host_norm && ps->host_output && !getenv("DOE_NO_HEADFOLD") &&
                     (ps->host_output_dt==12 || ps->host_output_dt==14);
            if (hf) {
                DOE_METAL_TRY(resident_fail, nt_metal_slot_alloc(SLOT_LOGITS, (uint64_t)ps->host_vocab*4));
                DOE_METAL_TRY(resident_fail, nt_metal_rmsnorm(SLOT_FN, SLOT_X, ps->host_norm, D, ps->rms_norm_eps));
                if (ps->host_output_dt==14)
                    { DOE_METAL_TRY(resident_fail, nt_metal_q6k_matvec_slot(SLOT_LOGITS, (const uint8_t*)ps->host_output, SLOT_FN, ps->host_vocab, D)); }
                else
                    { DOE_METAL_TRY(resident_fail, nt_metal_q4k_matvec_slot(SLOT_LOGITS, (const uint8_t*)ps->host_output, SLOT_FN, ps->host_vocab, D)); }
            }
            DOE_METAL_TRY(resident_fail, nt_metal_batch_commit());
            if (hf) { DOE_METAL_TRY(resident_fail, nt_metal_slot_download(SLOT_LOGITS, s->logits, (uint64_t)ps->host_vocab*4)); head_done = 1; }
            else      { DOE_METAL_TRY(resident_fail, nt_metal_slot_download(SLOT_X, s->x, D*4)); }
            resident_done = 1;
            struct timespec _r1; clock_gettime(CLOCK_MONOTONIC, &_r1);
            g_resident_ns += (_r1.tv_sec-_r0.tv_sec)*1e9 + (_r1.tv_nsec-_r0.tv_nsec);
            if (getenv("DOE_DEBUG_TIMING") && pos == 0)
                fprintf(stderr, "[resident] engaged: whole token in 1 command buffer (%d layers, parliament=%s)\n",
                        ps->host_n_layers, parl ? "on" : "off");
resident_fail:
            if (metal_rc) doe_metal_slot_fail(ps, &metal_slots_disabled, "resident decode", metal_op, metal_rc);
        }
    }
#endif

    for (int l = 0; !resident_done && l < ps->host_n_layers && l < MAX_LAYERS; l++) {
        if (!ps->host_layers[l].wq) continue;

        /* ── Host attention ── */
        int attn_done = 0;
#ifdef USE_METAL
        /* M4 stage (в): attention half-layer on GPU slots — rmsnorm -> qkv matvecs
         * -> rope -> copy_to_region(KV) -> attn_decode(GQA) -> o-proj -> residual,
         * one sync. rope_f32 == doe apply_rope (NeoX pairs, theta^-2i/hd). Guarded
         * to no-bias archs (Mistral) with Q4_K/Q6_K qkv/wo. CPU fallback otherwise. */
        {
            int qd=ps->host_layers[l].wq_dt, kdt=ps->host_layers[l].wk_dt,
                vd=ps->host_layers[l].wv_dt, od=ps->host_layers[l].wo_dt;
            if (!metal_slots_disabled && !getenv("DOE_NO_SLOTS") && nt_metal_available() && ps->host_layers[l].attn_norm &&
                !ps->host_layers[l].bq && !ps->host_layers[l].bk && !ps->host_layers[l].bv &&
                (qd==12||qd==14)&&(kdt==12||kdt==14)&&(vd==12||vd==14)&&(od==12||od==14)) {
                int metal_rc = 0;
                const char *metal_op = NULL;
                int co2 = l * s->max_seq * kd + pos * kd;
                float *Kbase = s->key_cache   + (size_t)l * s->max_seq * kd;
                float *Vbase = s->value_cache + (size_t)l * s->max_seq * kd;
                int qn = ps->host_heads * hd;
                DOE_METAL_TRY(attn_slot_fail, nt_metal_slot_alloc(SLOT_X, D*4));
                DOE_METAL_TRY(attn_slot_fail, nt_metal_slot_alloc(SLOT_FN, D*4));
                DOE_METAL_TRY(attn_slot_fail, nt_metal_slot_alloc(SLOT_Q, qn*4));
                DOE_METAL_TRY(attn_slot_fail, nt_metal_slot_alloc(SLOT_K, kd*4));
                DOE_METAL_TRY(attn_slot_fail, nt_metal_slot_alloc(SLOT_V, kd*4));
                DOE_METAL_TRY(attn_slot_fail, nt_metal_slot_alloc(SLOT_O, qn*4));
                DOE_METAL_TRY(attn_slot_fail, nt_metal_slot_alloc(SLOT_XB, D*4));
                DOE_METAL_TRY(attn_slot_fail, nt_metal_slot_upload(SLOT_X, s->x, D*4));
                DOE_METAL_TRY(attn_slot_fail, nt_metal_batch_begin());
                DOE_METAL_TRY(attn_slot_fail, nt_metal_rmsnorm(SLOT_FN, SLOT_X, ps->host_layers[l].attn_norm, D, ps->rms_norm_eps));
                if (qd==14)  { DOE_METAL_TRY(attn_slot_fail, nt_metal_q6k_matvec_slot(SLOT_Q,(const uint8_t*)ps->host_layers[l].wq,SLOT_FN,qn,D)); }
                else         { DOE_METAL_TRY(attn_slot_fail, nt_metal_q4k_matvec_slot(SLOT_Q,(const uint8_t*)ps->host_layers[l].wq,SLOT_FN,qn,D)); }
                if (kdt==14) { DOE_METAL_TRY(attn_slot_fail, nt_metal_q6k_matvec_slot(SLOT_K,(const uint8_t*)ps->host_layers[l].wk,SLOT_FN,kd,D)); }
                else         { DOE_METAL_TRY(attn_slot_fail, nt_metal_q4k_matvec_slot(SLOT_K,(const uint8_t*)ps->host_layers[l].wk,SLOT_FN,kd,D)); }
                if (vd==14)  { DOE_METAL_TRY(attn_slot_fail, nt_metal_q6k_matvec_slot(SLOT_V,(const uint8_t*)ps->host_layers[l].wv,SLOT_FN,kd,D)); }
                else         { DOE_METAL_TRY(attn_slot_fail, nt_metal_q4k_matvec_slot(SLOT_V,(const uint8_t*)ps->host_layers[l].wv,SLOT_FN,kd,D)); }
                int rope_norm = g_rope_norm || arch_rope_norm(ps->host_arch);
                DOE_METAL_TRY(attn_slot_fail, nt_metal_rope(SLOT_Q, ps->host_heads,    hd, pos, ps->rope_theta, rope_norm));
                DOE_METAL_TRY(attn_slot_fail, nt_metal_rope(SLOT_K, ps->host_kv_heads, hd, pos, ps->rope_theta, rope_norm));
                DOE_METAL_TRY(attn_slot_fail, nt_metal_copy_to_region(s->key_cache   + co2, SLOT_K, kd*4));
                DOE_METAL_TRY(attn_slot_fail, nt_metal_copy_to_region(s->value_cache + co2, SLOT_V, kd*4));
                DOE_METAL_TRY(attn_slot_fail, nt_metal_attn_decode(SLOT_O, SLOT_Q, Kbase, Vbase, pos+1,
                                                                   ps->host_heads, ps->host_kv_heads, hd,
                                                                   (uint32_t)kd, (uint32_t)hd, (uint32_t)kd, (uint32_t)hd, sc));
                if (od==14)  { DOE_METAL_TRY(attn_slot_fail, nt_metal_q6k_matvec_slot(SLOT_XB,(const uint8_t*)ps->host_layers[l].wo,SLOT_O,D,qn)); }
                else         { DOE_METAL_TRY(attn_slot_fail, nt_metal_q4k_matvec_slot(SLOT_XB,(const uint8_t*)ps->host_layers[l].wo,SLOT_O,D,qn)); }
                DOE_METAL_TRY(attn_slot_fail, nt_metal_add(SLOT_X, SLOT_X, SLOT_XB, D));
                DOE_METAL_TRY(attn_slot_fail, nt_metal_batch_commit());
                DOE_METAL_TRY(attn_slot_fail, nt_metal_slot_download(SLOT_X, s->x, D*4));
                attn_done = 1;
attn_slot_fail:
                if (metal_rc) doe_metal_slot_fail(ps, &metal_slots_disabled, "attention slot path", metal_op, metal_rc);
            }
        }
#endif
        if (!attn_done) {
        float *xn = s->xb;
        if (ps->host_layers[l].attn_norm) rmsnorm(xn, s->x, ps->host_layers[l].attn_norm, D, ps->rms_norm_eps);
        else memcpy(xn, s->x, D*4);

        /* {q,k,v} share xn — batch into one command buffer, one GPU sync (Metal).
         * out (s->q/k/v) is written only at commit, so commit must precede the
         * bias adds below that read them. Bit-identical to the solo path. */
#ifdef USE_METAL
        int qkv_batch = !getenv("DOE_NO_BATCH") && nt_metal_available() && nt_metal_batch_begin() == 0;
#endif
        doe_mv(s->q, ps->host_layers[l].wq, ps->host_layers[l].wq_dt, xn, ps->host_heads*hd, D);
        doe_mv(s->k, ps->host_layers[l].wk, ps->host_layers[l].wk_dt, xn, kd, D);
        doe_mv(s->v, ps->host_layers[l].wv, ps->host_layers[l].wv_dt, xn, kd, D);
#ifdef USE_METAL
        if (qkv_batch) nt_metal_batch_commit();
#endif

        /* Add attention biases (Qwen2, optional) */
        if (ps->host_layers[l].bq) for (int i = 0; i < ps->host_heads*hd; i++) s->q[i] += ps->host_layers[l].bq[i];
        if (ps->host_layers[l].bk) for (int i = 0; i < kd; i++) s->k[i] += ps->host_layers[l].bk[i];
        if (ps->host_layers[l].bv) for (int i = 0; i < kd; i++) s->v[i] += ps->host_layers[l].bv[i];

        int rope_norm = g_rope_norm || arch_rope_norm(ps->host_arch);
        for (int h = 0; h < ps->host_heads; h++) apply_rope_mode(s->q+h*hd, pos, s->cos_cache, s->sin_cache, hd, rope_norm);
        for (int h = 0; h < ps->host_kv_heads; h++) apply_rope_mode(s->k+h*hd, pos, s->cos_cache, s->sin_cache, hd, rope_norm);

        int co = l * s->max_seq * kd + pos * kd;
        memcpy(s->key_cache + co, s->k, kd*4);
        memcpy(s->value_cache + co, s->v, kd*4);

        float *ao = s->xb2; memset(ao, 0, D*4);
        for (int h = 0; h < ps->host_heads; h++) {
            int kvh = h / hg; float *qh = s->q + h*hd;
            float *att = s->att + h * s->max_seq;
            for (int t = 0; t <= pos; t++) {
                int ko = l*s->max_seq*kd + t*kd + kvh*hd;
                float dot = 0;
                for (int d = 0; d < hd; d++) dot += qh[d] * s->key_cache[ko+d];
                att[t] = dot * sc;
            }
            softmax_n(att, pos+1);
            float *oh = ao + h*hd;
            for (int t = 0; t <= pos; t++) {
                float a = att[t]; int vo = l*s->max_seq*kd + t*kd + kvh*hd;
                for (int d = 0; d < hd; d++) oh[d] += a * s->value_cache[vo+d];
            }
        }
        doe_mv(s->xb, ps->host_layers[l].wo, ps->host_layers[l].wo_dt, ao, D, ps->host_heads*hd); /* Y-1b: WO cols = heads*head_dim (4096), not dim (5120) */
        for (int i = 0; i < D; i++) s->x[i] += s->xb[i];
        } /* !attn_done */

        /* ── Parliament election + LoRA injection (after attention, before FFN) ── */
        /* A13: skip injection entirely when alpha=0 — avoids 0*NaN poisoning from drifted experts */
        if (l < ps->n_field_layers && ps->lora_alpha != 0.0f) {
            FieldLayer *fl = &ps->field_layers[l];
            int selected[MAX_EXPERTS]; float weights[MAX_EXPERTS];
            int k = parliament_elect(&fl->parliament, fl->experts, s->x, D, &s->hs, selected, weights);
            memset(s->expert_out, 0, D*4);
            for (int ki = 0; ki < k; ki++) {
                LoraExpert *exp = &fl->experts[selected[ki]];
                exp->tokens_seen++;
                /* Delta Voice: out += α × A @ (B @ x) */
                float tmp[LORA_RANK]; memset(tmp, 0, sizeof(tmp));
                for (int r = 0; r < ps->lora_rank; r++)
                    for (int j = 0; j < D; j++)
                        tmp[r] += exp->lora_B[r * D + j] * s->x[j];
                float lora_out[D]; memset(lora_out, 0, D*4);
                for (int i = 0; i < D; i++)
                    for (int r = 0; r < ps->lora_rank; r++)
                        lora_out[i] += exp->lora_A[i * ps->lora_rank + r] * tmp[r];
                for (int i = 0; i < D; i++)
                    s->expert_out[i] += weights[ki] * ps->lora_alpha * lora_out[i];
            }
            for (int i = 0; i < D; i++) s->x[i] += s->expert_out[i];
        }

        /* ── Host FFN (SwiGLU) ── */
        {
            int ffn_done = 0;
#ifdef USE_METAL
            /* M4 stage (b): FFN half-layer on GPU slots — rmsnorm -> gate/up matvecs ->
             * silu_mul -> down -> residual, one sync. Fused gate_up is split into two
             * sub-W matvecs (silu_mul needs two slots). x is host-bridged this stage
             * (attention still CPU); stage (г) keeps x resident. Falls through to the
             * CPU path when Metal is off or a dt is unsupported by the slot matvec. */
            {
                int gdt = ps->host_layers[l].ffn_gate_up_dt, ddt = ps->host_layers[l].ffn_down_dt;
                if (!metal_slots_disabled && !getenv("DOE_NO_SLOTS") && nt_metal_available() && ps->host_layers[l].ffn_norm &&
                    ps->host_layers[l].ffn_gate_up && ps->host_layers[l].ffn_down &&
                    (gdt==12||gdt==14) && (ddt==12||ddt==14)) {
                    int metal_rc = 0;
                    const char *metal_op = NULL;
                    const uint8_t *Wgu = (const uint8_t*)ps->host_layers[l].ffn_gate_up;
                    const uint8_t *Wdn = (const uint8_t*)ps->host_layers[l].ffn_down;
                    size_t gu_row = quant_raw_bytes(gdt, D);
                    DOE_METAL_TRY(ffn_slot_fail, nt_metal_slot_alloc(SLOT_X, D*4));
                    DOE_METAL_TRY(ffn_slot_fail, nt_metal_slot_alloc(SLOT_FN, D*4));
                    DOE_METAL_TRY(ffn_slot_fail, nt_metal_slot_alloc(SLOT_G, H*4));
                    DOE_METAL_TRY(ffn_slot_fail, nt_metal_slot_alloc(SLOT_U, H*4));
                    DOE_METAL_TRY(ffn_slot_fail, nt_metal_slot_alloc(SLOT_HB, H*4));
                    DOE_METAL_TRY(ffn_slot_fail, nt_metal_slot_alloc(SLOT_DN, D*4));
                    DOE_METAL_TRY(ffn_slot_fail, nt_metal_slot_upload(SLOT_X, s->x, D*4));
                    DOE_METAL_TRY(ffn_slot_fail, nt_metal_batch_begin());
                    DOE_METAL_TRY(ffn_slot_fail, nt_metal_rmsnorm(SLOT_FN, SLOT_X, ps->host_layers[l].ffn_norm, D, ps->rms_norm_eps));
                    if (gdt==14) { DOE_METAL_TRY(ffn_slot_fail, nt_metal_q6k_matvec_slot(SLOT_G, Wgu, SLOT_FN, H, D));
                                   DOE_METAL_TRY(ffn_slot_fail, nt_metal_q6k_matvec_slot(SLOT_U, Wgu+(size_t)H*gu_row, SLOT_FN, H, D)); }
                    else         { DOE_METAL_TRY(ffn_slot_fail, nt_metal_q4k_matvec_slot(SLOT_G, Wgu, SLOT_FN, H, D));
                                   DOE_METAL_TRY(ffn_slot_fail, nt_metal_q4k_matvec_slot(SLOT_U, Wgu+(size_t)H*gu_row, SLOT_FN, H, D)); }
                    DOE_METAL_TRY(ffn_slot_fail, nt_metal_silu_mul(SLOT_HB, SLOT_G, SLOT_U, H));
                    if (ddt==14) { DOE_METAL_TRY(ffn_slot_fail, nt_metal_q6k_matvec_slot(SLOT_DN, Wdn, SLOT_HB, D, H)); }
                    else         { DOE_METAL_TRY(ffn_slot_fail, nt_metal_q4k_matvec_slot(SLOT_DN, Wdn, SLOT_HB, D, H)); }
                    DOE_METAL_TRY(ffn_slot_fail, nt_metal_add(SLOT_X, SLOT_X, SLOT_DN, D));
                    DOE_METAL_TRY(ffn_slot_fail, nt_metal_batch_commit());
                    DOE_METAL_TRY(ffn_slot_fail, nt_metal_slot_download(SLOT_X, s->x, D*4));
                    ffn_done = 1;
ffn_slot_fail:
                    if (metal_rc) doe_metal_slot_fail(ps, &metal_slots_disabled, "FFN slot path", metal_op, metal_rc);
                }
            }
#endif
            if (!ffn_done) {
            float *fn = s->xb;
            if (ps->host_layers[l].ffn_norm) rmsnorm(fn, s->x, ps->host_layers[l].ffn_norm, D, ps->rms_norm_eps);
            else memcpy(fn, s->x, D*4);

            if (ps->host_layers[l].ffn_gate_up && ps->host_layers[l].ffn_down) {
                /* Fused gate_up: [hidden*2, dim] → split into gate [0..H) and up [H..2H) */
                doe_mv(s->hb2, ps->host_layers[l].ffn_gate_up, ps->host_layers[l].ffn_gate_up_dt, fn, H * 2, D);
                for (int i = 0; i < H; i++) s->hb[i] = silu_f(s->hb2[i]) * s->hb2[H + i];
                doe_mv(s->xb, ps->host_layers[l].ffn_down, ps->host_layers[l].ffn_down_dt, s->hb, D, H);
                for (int i = 0; i < D; i++) s->x[i] += s->xb[i];
            } else if (ps->host_layers[l].ffn_gate && ps->host_layers[l].ffn_up && ps->host_layers[l].ffn_down) {
                /* Standard separate gate + up — share fn, batch into one sync. */
#ifdef USE_METAL
                int gu_batch = !getenv("DOE_NO_BATCH") && nt_metal_available() && nt_metal_batch_begin() == 0;
#endif
                doe_mv(s->hb, ps->host_layers[l].ffn_gate, ps->host_layers[l].ffn_gate_dt, fn, H, D);
                doe_mv(s->hb2, ps->host_layers[l].ffn_up, ps->host_layers[l].ffn_up_dt, fn, H, D);
#ifdef USE_METAL
                if (gu_batch) nt_metal_batch_commit();
#endif
                for (int i = 0; i < H; i++) s->hb[i] = silu_f(s->hb[i]) * s->hb2[i];
                doe_mv(s->xb, ps->host_layers[l].ffn_down, ps->host_layers[l].ffn_down_dt, s->hb, D, H);
                for (int i = 0; i < D; i++) s->x[i] += s->xb[i];
            }
            } /* !ffn_done */
        }
    }

    /* Final norm + LM head (skipped when folded into the resident CB above) */
    if (!head_done) {
        struct timespec _h0; clock_gettime(CLOCK_MONOTONIC, &_h0);
        rmsnorm(s->x, s->x, ps->host_norm, D, ps->rms_norm_eps);
        doe_mv(s->logits, ps->host_output, ps->host_output_dt, s->x, ps->host_vocab, D);
        struct timespec _h1; clock_gettime(CLOCK_MONOTONIC, &_h1);
        g_head_ns += (_h1.tv_sec-_h0.tv_sec)*1e9 + (_h1.tv_nsec-_h0.tv_nsec);
    }

    return s->logits;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SAMPLING + CHAT
 * ═══════════════════════════════════════════════════════════════════════════════ */
static int sample(float *logits, int V, float temp, int top_k) {
    if (!logits || V <= 0) return 0;
    int finite = 0, nonfinite = 0;
    for (int i = 0; i < V; i++) {
        if (isfinite(logits[i])) finite++;
        else { logits[i] = -1e30f; nonfinite++; }
    }
    if (nonfinite && !g_sampler_nonfinite_warned) {
        fprintf(stderr, "[doe] WARNING: sampler replaced non-finite logits with floor values\n");
        g_sampler_nonfinite_warned = 1;
    }
    if (finite == 0) {
        fprintf(stderr, "[doe] sampler saw no finite logits; returning token 0\n");
        return 0;
    }
    if (!isfinite(temp)) temp = 1.0f;
    if (temp <= 0) { int b = 0; for (int i = 1; i < V; i++) if (logits[i] > logits[b]) b = i; return b; }
    for (int i = 0; i < V; i++) logits[i] /= temp;
    if (top_k > 0 && top_k < V) {
        /* threshold = top_k-th largest logit via a size-k min-heap: O(V log k),
         * no per-token malloc. Root holds the smallest of the k kept = same
         * threshold the old O(k*V) selection-sort produced (value is identical
         * regardless of tie-breaking), so the sampled token is bit-identical. */
        int hk = top_k;
        if (hk > 256) {
            if (!g_top_k_clamp_warned) {
                fprintf(stderr, "[doe] WARNING: --top-k %d exceeds sampler heap cap 256; clamping\n", top_k);
                g_top_k_clamp_warned = 1;
            }
            hk = 256;
        }
        float heap[256]; int n = 0;
        for (int i = 0; i < V; i++) {
            float v = logits[i];
            if (n < hk) {
                int c = n++; heap[c] = v;
                while (c > 0) { int p=(c-1)/2; if (heap[p] <= heap[c]) break;
                               float t=heap[p]; heap[p]=heap[c]; heap[c]=t; c=p; }
            } else if (v > heap[0]) {
                heap[0] = v; int c = 0;
                for (;;) { int l=2*c+1, r=2*c+2, m=c;
                           if (l<hk && heap[l]<heap[m]) m=l;
                           if (r<hk && heap[r]<heap[m]) m=r;
                           if (m==c) break;
                           float t=heap[m]; heap[m]=heap[c]; heap[c]=t; c=m; }
            }
        }
        float th = heap[0];
        for (int i = 0; i < V; i++) if (logits[i] < th) logits[i] = -1e30f;
    }
    softmax_n(logits, V);
    float r = rand_uniform(), cum = 0;
    for (int i = 0; i < V; i++) { cum += logits[i]; if (cum >= r) return i; }
    return V - 1;
}

/* GPT-2 byte_decoder: reverse the byte_encoder mapping (unicode codepoint -> original byte) */
static int gpt2_rune_to_byte(int rune) {
    static int table_built = 0;
    static int rtable[512]; /* rune -> byte, -1 if not mapped */
    if (!table_built) {
        for (int i = 0; i < 512; i++) rtable[i] = -1;
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255))
                rtable[b] = b; /* identity mapping */
            else
                rtable[256 + n++] = b; /* offset mapping */
        }
        table_built = 1;
    }
    if (rune >= 0 && rune < 512) return rtable[rune];
    return -1;
}

/* Parse one UTF-8 codepoint, return codepoint and advance *p by bytes consumed */
static int utf8_decode_cp(const char **p) {
    const unsigned char *s = (const unsigned char *)*p;
    int cp, len;
    if (s[0] < 0x80) { cp = s[0]; len = 1; }
    else if ((s[0] & 0xE0) == 0xC0) { cp = (s[0] & 0x1F) << 6 | (s[1] & 0x3F); len = 2; }
    else if ((s[0] & 0xF0) == 0xE0) { cp = (s[0] & 0x0F) << 12 | (s[1] & 0x3F) << 6 | (s[2] & 0x3F); len = 3; }
    else if ((s[0] & 0xF8) == 0xF0) { cp = (s[0] & 0x07) << 18 | (s[1] & 0x3F) << 12 | (s[2] & 0x3F) << 6 | (s[3] & 0x3F); len = 4; }
    else { cp = s[0]; len = 1; } /* fallback */
    *p += len;
    return cp;
}

/* Decode token to text using GGUF vocab, fallback to byte */
static void token_decode_print(GGUFIndex *ps, int token) {
    if (ps->vocab_tokens && token >= 0 && token < ps->vocab_size && ps->vocab_tokens[token]) {
        const char *s = ps->vocab_tokens[token];
        if (ps->is_gpt2_bpe) {
            /* GPT-2 byte-level BPE: full byte_decoder — each codepoint maps to one byte */
            unsigned char buf[256];
            int blen = 0;
            const char *p = s;
            while (*p && blen < (int)sizeof(buf) - 4) {
                int cp = utf8_decode_cp(&p);
                int b = gpt2_rune_to_byte(cp);
                if (b >= 0) buf[blen++] = (unsigned char)b;
                else {
                    /* Not a GPT-2 mapped byte — emit codepoint as UTF-8 */
                    if (cp < 0x80) { buf[blen++] = cp; }
                    else if (cp < 0x800) { buf[blen++] = 0xC0|(cp>>6); buf[blen++] = 0x80|(cp&0x3F); }
                    else if (cp < 0x10000) { buf[blen++] = 0xE0|(cp>>12); buf[blen++] = 0x80|((cp>>6)&0x3F); buf[blen++] = 0x80|(cp&0x3F); }
                    else { buf[blen++] = 0xF0|(cp>>18); buf[blen++] = 0x80|((cp>>12)&0x3F); buf[blen++] = 0x80|((cp>>6)&0x3F); buf[blen++] = 0x80|(cp&0x3F); }
                }
            }
            fwrite(buf, 1, blen, stdout);
        } else {
        /* Handle sentencepiece ▁ (U+2581, 3 bytes: E2 96 81) → space */
        while (*s) {
            if ((unsigned char)s[0] == 0xE2 && (unsigned char)s[1] == 0x96 && (unsigned char)s[2] == 0x81) {
                fputc(' ', stdout);
                s += 3;
            } else if (!strncmp(s, "<0x", 3) && s[5] == '>') {
                /* sentencepiece hex byte: <0xAB> */
                unsigned int b = 0;
                sscanf(s + 3, "%02X", &b);
                if (b >= 32 || b == '\n' || b == '\t') fputc((char)b, stdout);
                s += 6;
            } else {
                fputc(*s, stdout);
                s++;
            }
        }
        }
    } else if (token >= 0 && token < 256) {
        char c = (char)token;
        if (c >= 32 || c == '\n' || c == '\t') fputc(c, stdout);
    }
}

/* Decode token to buffer instead of stdout — for HTTP serve mode */
static int token_decode_buf(GGUFIndex *ps, int token, char *buf, int bufsz) {
    int pos = 0;
    if (ps->vocab_tokens && token >= 0 && token < ps->vocab_size && ps->vocab_tokens[token]) {
        const char *s = ps->vocab_tokens[token];
        if (ps->is_gpt2_bpe) {
            const char *p = s;
            while (*p && pos < bufsz - 4) {
                int cp = utf8_decode_cp(&p);
                int b = gpt2_rune_to_byte(cp);
                if (b >= 0) buf[pos++] = (char)(unsigned char)b;
                else {
                    if (cp < 0x80) { buf[pos++] = cp; }
                    else if (cp < 0x800 && pos < bufsz-2) { buf[pos++] = 0xC0|(cp>>6); buf[pos++] = 0x80|(cp&0x3F); }
                    else if (cp < 0x10000 && pos < bufsz-3) { buf[pos++] = 0xE0|(cp>>12); buf[pos++] = 0x80|((cp>>6)&0x3F); buf[pos++] = 0x80|(cp&0x3F); }
                }
            }
        } else {
            while (*s && pos < bufsz - 1) {
                if ((unsigned char)s[0]==0xE2 && (unsigned char)s[1]==0x96 && (unsigned char)s[2]==0x81) {
                    buf[pos++]=' '; s+=3;
                } else if (!strncmp(s,"<0x",3) && s[5]=='>') {
                    unsigned int b=0; sscanf(s+3,"%02X",&b);
                    if (b>=32||b=='\n'||b=='\t') buf[pos++]=(char)b;
                    s+=6;
                } else { buf[pos++]=*s; s++; }
            }
        }
    } else if (token >= 0 && token < 256) {
        char c = (char)token;
        if ((c>=32||c=='\n'||c=='\t') && pos < bufsz-1) buf[pos++]=c;
    }
    buf[pos] = '\0';
    return pos;
}

/* ── BPE Tokenizer — SentencePiece style, score-based merge ── */

static int tok_lookup(GGUFIndex *ps, const char *s, int len); /* forward decl */

/* Build GPT-2 BPE scores from merges (called after index_load if needed) */
static void build_gpt2_scores(GGUFIndex *ps) {
    if (!ps || !ps->is_gpt2_bpe || !ps->bpe_merges || ps->n_bpe_merges <= 0 ||
        ps->vocab_scores || !ps->vocab_tokens || ps->vocab_size <= 0) return;
    ps->vocab_scores = calloc(ps->vocab_size, sizeof(float));
    if (!ps->vocab_scores) return;
    ps->vocab_scores_size = ps->vocab_size;
    for (int i = 0; i < ps->vocab_size; i++) ps->vocab_scores[i] = -1e9f;
    int built = 0;
    for (int m = 0; m < ps->n_bpe_merges; m++) {
        const char *merge = ps->bpe_merges[m];
        const char *sp = strchr(merge, ' ');
        if (!sp) continue;
        int la = (int)(sp - merge), lb = (int)strlen(sp + 1);
        if (la + lb > 128) continue;
        char merged[130];
        memcpy(merged, merge, la);
        memcpy(merged + la, sp + 1, lb);
        int mid = tok_lookup(ps, merged, la + lb);
        if (mid >= 0 && mid < ps->vocab_scores_size) {
            ps->vocab_scores[mid] = (float)(ps->n_bpe_merges - m);
            built++;
        }
    }
    printf("[doe] GPT-2 BPE: built %d merge scores from %d merges\n", built, ps->n_bpe_merges);
    ps->add_space_prefix = 0;
}

/* FNV-1a hash */
static uint32_t tok_hash(const char *s, int len) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) { h ^= (uint8_t)s[i]; h *= 16777619u; }
    return h;
}

/* Build hash table for O(1) token lookup */
static void tok_ht_build(GGUFIndex *ps) {
    if (!ps || !ps->vocab_tokens || ps->vocab_size <= 0) return;
    int cap = 1;
    while (cap < ps->vocab_size * 3) cap <<= 1; /* ~33% load factor */
    ps->tok_ht_ids = malloc(cap * sizeof(int));
    if (!ps->tok_ht_ids) {
        ps->tok_ht_cap = 0;
        fprintf(stderr, "[doe] tokenizer hash table allocation failed; falling back to linear lookup\n");
        return;
    }
    ps->tok_ht_cap = cap;
    for (int i = 0; i < cap; i++) ps->tok_ht_ids[i] = -1;
    int mask = cap - 1;
    for (int i = 0; i < ps->vocab_size; i++) {
        if (!ps->vocab_tokens[i]) continue;
        int slen = (int)strlen(ps->vocab_tokens[i]);
        uint32_t idx = tok_hash(ps->vocab_tokens[i], slen) & mask;
        while (ps->tok_ht_ids[idx] != -1) idx = (idx + 1) & mask;
        ps->tok_ht_ids[idx] = i;
    }
}

/* Find token ID by string. Returns -1 if not found. O(1) average. */
static int tok_lookup(GGUFIndex *ps, const char *s, int len) {
    if (!ps || !s || len <= 0 || !ps->vocab_tokens || ps->vocab_size <= 0) return -1;
    if (!ps->tok_ht_ids) {
        /* fallback linear scan */
        for (int i = 0; i < ps->vocab_size; i++) {
            if (ps->vocab_tokens[i] && strlen(ps->vocab_tokens[i]) == (size_t)len
                && memcmp(ps->vocab_tokens[i], s, len) == 0)
                return i;
        }
        return -1;
    }
    if (ps->tok_ht_cap <= 0) return -1;
    int mask = ps->tok_ht_cap - 1;
    uint32_t idx = tok_hash(s, len) & mask;
    while (ps->tok_ht_ids[idx] != -1) {
        int id = ps->tok_ht_ids[idx];
        if (id < 0 || id >= ps->vocab_size) return -1;
        const char *t = ps->vocab_tokens[id];
        if (t && strlen(t) == (size_t)len && memcmp(t, s, len) == 0) return id;
        idx = (idx + 1) & mask;
    }
    return -1;
}

/* Score-based BPE merge on an array of token IDs */
static int bpe_merge(GGUFIndex *ps, int *ids, int n) {
    if (!ps || !ids || n <= 1 || !ps->vocab_scores || !ps->vocab_tokens ||
        ps->vocab_size <= 0 || ps->vocab_scores_size <= 0) return n;
    while (n > 1) {
        float best_score = -1e30f;
        int best_idx = -1, best_id = -1;
        for (int i = 0; i < n - 1; i++) {
            /* Concatenate token strings */
            int a_id = ids[i], b_id = ids[i+1];
            if (a_id < 0 || a_id >= ps->vocab_size ||
                b_id < 0 || b_id >= ps->vocab_size) continue;
            const char *a = ps->vocab_tokens[a_id];
            const char *b = ps->vocab_tokens[b_id];
            if (!a || !b) continue;
            int la = strlen(a), lb = strlen(b);
            if (la + lb > 128) continue;
            char merged[130];
            memcpy(merged, a, la);
            memcpy(merged + la, b, lb);
            int mid = tok_lookup(ps, merged, la + lb);
            if (mid >= 0 && mid < ps->vocab_size && mid < ps->vocab_scores_size &&
                ps->vocab_scores[mid] > best_score) {
                best_score = ps->vocab_scores[mid];
                best_idx = i;
                best_id = mid;
            }
        }
        if (best_idx < 0) break;
        ids[best_idx] = best_id;
        /* Remove ids[best_idx+1] by shifting */
        for (int i = best_idx + 1; i < n - 1; i++) ids[i] = ids[i+1];
        n--;
    }
    return n;
}

/* GPT-2 byte-to-unicode table: maps each byte to a unicode codepoint */
static int gpt2_byte_to_rune(int b) {
    /* Printable ASCII + Latin-1 supplement range → identity */
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255))
        return b;
    /* Everything else → 256 + offset */
    static int table_built = 0;
    static int table[256];
    if (!table_built) {
        int n = 0;
        for (int i = 0; i < 256; i++) {
            if ((i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174 && i <= 255))
                table[i] = i;
            else
                table[i] = 256 + n++;
        }
        table_built = 1;
    }
    return table[(unsigned char)b];
}

/* Encode a unicode codepoint as UTF-8, return length */
static int rune_to_utf8(int r, char *out) {
    if (r < 0x80) { out[0] = (char)r; return 1; }
    if (r < 0x800) { out[0] = 0xC0 | (r >> 6); out[1] = 0x80 | (r & 0x3F); return 2; }
    out[0] = 0xE0 | (r >> 12); out[1] = 0x80 | ((r >> 6) & 0x3F); out[2] = 0x80 | (r & 0x3F); return 3;
}

/* Try to match a special token at position i in text. Returns token id and advances *len. */
static int try_special_token(GGUFIndex *ps, const char *text, int tlen, int i, int *consumed) {
    static const char *specials[] = {
        "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|end|>",
        "<start_of_turn>", "<end_of_turn>", "<|user|>", "<|assistant|>",
        "[INST]", "[/INST]", "<s>", "</s>",
        "<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>",
        "<|bos|>", "<|eot_id|>", NULL
    };
    if (text[i] != '<' && text[i] != '[') return -1;
    for (int s = 0; specials[s]; s++) {
        int slen = (int)strlen(specials[s]);
        if (i + slen <= tlen && memcmp(text + i, specials[s], slen) == 0) {
            int id = tok_lookup(ps, specials[s], slen);
            if (id >= 0) { *consumed = slen; return id; }
        }
    }
    return -1;
}

static void warn_tokenizer_zero_fallback(const char *kind, unsigned char byte) {
    if (!g_tokenizer_zero_fallback_warned) {
        fprintf(stderr, "[doe] WARNING: tokenizer missing %s fallback for byte 0x%02X; using token 0\n",
                kind ? kind : "byte", (unsigned int)byte);
        g_tokenizer_zero_fallback_warned = 1;
    }
}

static int tokenize_input(GGUFIndex *ps, const char *text, int *tokens, int max_tokens) {
    if (!text || !tokens || max_tokens <= 0) return 0;
    if (!ps->vocab_tokens) {
        int n = 0, len = strlen(text), i = 0;
        for (; i < len && n < max_tokens; i++) tokens[n++] = (unsigned char)text[i];
        if (i < len) {
            fprintf(stderr, "[doe] prompt tokenization exceeds token budget (%d tokens)\n", max_tokens);
            return -1;
        }
        return n;
    }

    int tlen = strlen(text);
    if (tlen > (INT32_MAX - 16) / 3) {
        fprintf(stderr, "[doe] prompt too large to tokenize safely\n");
        return -1;
    }
    int ids_cap = tlen * 3 + 16; /* SentencePiece ▁ fallback can emit up to 3 ids/input byte. */
    int *ids = malloc((size_t)ids_cap * sizeof(int));
    if (!ids) {
        fprintf(stderr, "[doe] tokenizer allocation failed for %d ids\n", ids_cap);
        return -1;
    }
    int n = 0;
    int truncated = 0;

    if (ps->is_gpt2_bpe) {
        /* GPT-2: check special tokens first, then byte-level BPE */
        for (int i = 0; i < tlen; ) {
            if (n >= ids_cap) { truncated = 1; break; }
            int consumed = 0;
            int sid = try_special_token(ps, text, tlen, i, &consumed);
            if (sid >= 0) { ids[n++] = sid; i += consumed; continue; }
            int r = gpt2_byte_to_rune((unsigned char)text[i]);
            char u8[4]; int u8len = rune_to_utf8(r, u8);
            int id = tok_lookup(ps, u8, u8len);
            if (id >= 0) ids[n++] = id;
            else {
                warn_tokenizer_zero_fallback("GPT-2 byte", (unsigned char)text[i]);
                ids[n++] = 0;
            }
            i++;
        }
    } else {
        /* SentencePiece: split on special tokens first, then ▁-encode segments */
        int i = 0;
        while (i < tlen) {
            if (n >= ids_cap) { truncated = 1; break; }
            /* Check special tokens at raw text level */
            int consumed = 0;
            int sid = try_special_token(ps, text, tlen, i, &consumed);
            if (sid >= 0) { ids[n++] = sid; i += consumed; continue; }

            /* Find next special token boundary (or end) */
            int seg_end = i + 1;
            while (seg_end < tlen) {
                int c2 = 0;
                if (try_special_token(ps, text, tlen, seg_end, &c2) >= 0) break;
                seg_end++;
            }

            /* Encode segment [i, seg_end) with SentencePiece ▁ */
            int slen = seg_end - i;
            char *sp = malloc(slen * 3 + 4);
            if (!sp) {
                fprintf(stderr, "[doe] tokenizer segment allocation failed\n");
                free(ids);
                return -1;
            }
            int sp_len = 0;
            if (ps->add_space_prefix && i == 0 && text[i] != ' ') {
                sp[sp_len++] = 0xE2; sp[sp_len++] = 0x96; sp[sp_len++] = 0x81;
            }
            for (int j = i; j < seg_end; j++) {
                if (text[j] == ' ') {
                    sp[sp_len++] = 0xE2; sp[sp_len++] = 0x96; sp[sp_len++] = 0x81;
                } else {
                    sp[sp_len++] = text[j];
                }
            }
            sp[sp_len] = '\0';
            int k = 0;
            while (k < sp_len) {
                if (n >= ids_cap) { truncated = 1; break; }
                int clen = 1;
                unsigned char c = (unsigned char)sp[k];
                if (c >= 0xC0 && c < 0xE0) clen = 2;
                else if (c >= 0xE0 && c < 0xF0) clen = 3;
                else if (c >= 0xF0) clen = 4;
                if (k + clen > sp_len) clen = 1;
                int id = tok_lookup(ps, sp + k, clen);
                if (id >= 0) { ids[n++] = id; k += clen; }
                else {
                    char hex[7]; snprintf(hex, 7, "<0x%02X>", (unsigned char)sp[k]);
                    id = tok_lookup(ps, hex, 6);
                    if (id >= 0) ids[n++] = id;
                    else {
                        warn_tokenizer_zero_fallback("SentencePiece byte", (unsigned char)sp[k]);
                        ids[n++] = 0;
                    }
                    k++;
                }
            }
            free(sp);
            if (truncated) break;
            i = seg_end;
        }
    }
    if (truncated) {
        fprintf(stderr, "[doe] tokenizer workspace exhausted (%d ids)\n", ids_cap);
        free(ids);
        return -1;
    }

    n = bpe_merge(ps, ids, n);
    if (n > max_tokens) {
        fprintf(stderr, "[doe] prompt tokenization exceeds token budget (%d > %d tokens)\n",
                n, max_tokens);
        free(ids);
        return -1;
    }
    int out = n;
    memcpy(tokens, ids, out * sizeof(int));
    free(ids);
    return out;
}

static int tokenize_append_checked(GGUFIndex *ps, const char *text, int *tokens,
                                   int token_cap, int *n_tokens, const char *what) {
    if (!n_tokens || *n_tokens < 0 || *n_tokens > token_cap) return 0;
    int avail = token_cap - *n_tokens;
    if (avail <= 0) {
        fprintf(stderr, "[doe] no token budget left for %s (cap %d)\n", what, token_cap);
        return 0;
    }
    int got = tokenize_input(ps, text, tokens + *n_tokens, avail);
    if (got < 0) {
        fprintf(stderr, "[doe] %s refused: token budget exceeded\n", what);
        return 0;
    }
    *n_tokens += got;
    return 1;
}

static int token_append_checked(int token, int *tokens, int token_cap, int *n_tokens, const char *what) {
    if (!tokens || !n_tokens || *n_tokens < 0 || *n_tokens >= token_cap) {
        fprintf(stderr, "[doe] no token budget left for %s (cap %d)\n", what, token_cap);
        return 0;
    }
    tokens[(*n_tokens)++] = token;
    return 1;
}

static float *load_img_embeds_bin(const char *path, int host_dim, int *o_ntok) {
    if (o_ntok) *o_ntok = 0;
    if (!path || host_dim <= 0 || !o_ntok) return NULL;
    errno = 0;
    FILE *ef = fopen(path, "rb");
    if (!ef) {
        int open_errno = errno;
        printf("[doe] cannot open img-embeds-bin %s%s%s\n",
               path, open_errno ? ": " : "", open_errno ? strerror(open_errno) : "");
        return NULL;
    }
    errno = 0;
    if (fseek(ef, 0, SEEK_END) != 0) {
        int seek_errno = errno;
        printf("[doe] cannot seek img-embeds-bin %s%s%s\n",
               path, seek_errno ? ": " : "", seek_errno ? strerror(seek_errno) : "");
        fclose(ef);
        return NULL;
    }
    errno = 0;
    long sz = ftell(ef);
    int ftell_errno = errno;
    if (sz < 0) {
        printf("[doe] cannot size img-embeds-bin %s%s%s\n",
               path, ftell_errno ? ": " : "", ftell_errno ? strerror(ftell_errno) : "");
        fclose(ef);
        return NULL;
    }
    errno = 0;
    if (fseek(ef, 0, SEEK_SET) != 0) {
        int rewind_errno = errno;
        printf("[doe] cannot rewind img-embeds-bin %s%s%s\n",
               path, rewind_errno ? ": " : "", rewind_errno ? strerror(rewind_errno) : "");
        fclose(ef);
        return NULL;
    }
    if (sz == 0 || (sz % (long)sizeof(float)) != 0) {
        printf("[doe] invalid img-embeds-bin byte size: %ld\n", sz);
        fclose(ef);
        return NULL;
    }
    long ne = sz / (long)sizeof(float);
    if ((ne % host_dim) != 0 || ne / host_dim > INT_MAX) {
        printf("[doe] img-embeds-bin shape mismatch: %ld floats for host_dim=%d\n", ne, host_dim);
        fclose(ef);
        return NULL;
    }
    int n_img = (int)(ne / host_dim);
    if (n_img <= 0) {
        printf("[doe] img-embeds-bin contains no image tokens\n");
        fclose(ef);
        return NULL;
    }
    if ((unsigned long)ne > SIZE_MAX / sizeof(float)) {
        printf("[doe] img-embeds-bin too large: %ld floats\n", ne);
        fclose(ef);
        return NULL;
    }
    errno = 0;
    float *emb = malloc((size_t)ne * sizeof(float));
    if (!emb) {
        int malloc_errno = errno;
        printf("[doe] img-embeds-bin allocation failed for %s: %ld floats%s%s\n",
               path, ne, malloc_errno ? ": " : "", malloc_errno ? strerror(malloc_errno) : "");
        fclose(ef);
        return NULL;
    }
    errno = 0;
    size_t got = fread(emb, sizeof(float), (size_t)ne, ef);
    int read_errno = errno;
    if (got != (size_t)ne) {
        if (ferror(ef)) {
            printf("[doe] read error from img-embeds-bin %s after %zu/%zu floats%s%s\n",
                   path, got, (size_t)ne,
                   read_errno ? ": " : "", read_errno ? strerror(read_errno) : "");
        } else {
            printf("[doe] short read from img-embeds-bin %s (%zu/%zu floats)\n",
                   path, got, (size_t)ne);
        }
        free(emb);
        fclose(ef);
        return NULL;
    }
    fclose(ef);
    *o_ntok = n_img;
    return emb;
}

static int doe_snprintf_checked(char *dst, size_t dstsz, const char *what, const char *fmt, ...) {
    if (!dst || dstsz == 0) return 0;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(dst, dstsz, fmt, ap);
    va_end(ap);
    if (n < 0) {
        dst[0] = '\0';
        fprintf(stderr, "[doe] %s formatting failed\n", what);
        return 0;
    }
    if ((size_t)n >= dstsz) {
        dst[0] = '\0';
        fprintf(stderr, "[doe] %s too long (%d bytes, cap %zu); refusing truncated prompt\n",
                what, n, dstsz - 1);
        return 0;
    }
    return 1;
}

static int doe_control_nonce_valid(const char *s) {
    if (!s || !*s) return 0;
    size_t len = strlen(s);
    if (len > 128) return 0;
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        if ((c >= '0' && c <= '9') ||
            (c >= 'A' && c <= 'Z') ||
            (c >= 'a' && c <= 'z') ||
            c == '-' || c == '_' || c == '.')
            continue;
        return 0;
    }
    return 1;
}

static void chat(GGUFIndex *ps) {
    int max_seq = 1536;  /* room for ~1022 image tokens / long contexts */
    InferState is = alloc_infer(ps, max_seq);
    if (!is.x) {
        fprintf(stderr, "[doe] cannot start chat: inference buffers unavailable\n");
        return;
    }
    CalendarDrift cd; drift_init(&cd);
    MetaTrack meta; meta_init(&meta);

    char input[8192];  /* enlarged for long prompts / long-context probes */
    printf("\n[doe] the parliament is in session. type your message (Ctrl+C to dissipate):\n");
    printf("[doe] host: %s (%s, %dM params)\n\n",
           ps->host_path, ps->host_arch,
           (int)(ps->host_n_params / 1000000));

    float debt_sum = 0; int debt_count = 0;

    while (1) {
        printf("> "); fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) break;
        int len = strlen(input);
        while (len > 0 && (input[len-1]=='\n' || input[len-1]=='\r')) input[--len] = '\0';
        if (!len) continue;
        if (strcmp(input,"quit")==0 || strcmp(input,"exit")==0) break;
        if (strncmp(input, "status ", 7) == 0 && doe_control_nonce_valid(input + 7)) {
            const char *nonce = input + 7;
            printf("[field-control] nonce=%s step=%d debt=%.3f entropy=%.3f resonance=%.3f emergence=%.3f\n",
                   nonce, F.step, F.debt, F.entropy, F.resonance, F.emergence);
            continue;
        }
        if (strcmp(input,"status")==0) {
            printf("[field] step=%d debt=%.3f entropy=%.3f resonance=%.3f emergence=%.3f\n",
                   F.step, F.debt, F.entropy, F.resonance, F.emergence);
            printf("[field] season=%s health=%.3f temp=%.3f velocity=%s\n",
                   (const char*[]){"spring","summer","autumn","winter"}[F.season],
                   F.field_health, F.effective_temp,
                   (const char*[]){"nomove","walk","run","backward"}[F.velocity_mode]);
            printf("[drift] d=%.3f stability=%.3f accel=%.4f snapshots=%d\n",
                   cd.drift, cd.stability, cd.drift_accel, cd.n_snapshots);
            int te = 0;
            for (int l = 0; l < ps->n_field_layers; l++) te += ps->field_layers[l].n_alive;
            printf("[experts] alive=%d consensus=%.2f elections=%d\n",
                   te, ps->field_layers[0].parliament.consensus,
                   ps->field_layers[0].parliament.election_count);
            if (debt_count > 0)
                printf("[prophecy] avg_debt=%.4f total_debt=%.4f\n", debt_sum/debt_count, F.debt);
            continue;
        }

        /* Reset KV cache */
        memset(is.key_cache, 0, is.kv_bytes);
        memset(is.value_cache, 0, is.kv_bytes);

        /* Wrap input in chat template (auto-detected from GGUF chat_template) */
        char wrapped[sizeof(input) + 128];
        /* Only use chat template if the key special tokens exist in vocab */
        int use_template = 0;
        switch (ps->chat_style) {
        case 1: /* ChatML */
            if (tok_lookup(ps, "<|im_start|>", 12) >= 0) {
                if (!doe_snprintf_checked(wrapped, sizeof(wrapped), "ChatML prompt",
                        "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", input))
                    continue;
                use_template = 1;
            }
            break;
        case 2: /* [INST] */
            if (tok_lookup(ps, "[INST]", 6) >= 0) {
                if (!doe_snprintf_checked(wrapped, sizeof(wrapped), "INST prompt", "[INST] %s [/INST]", input))
                    continue;
                use_template = 1;
            }
            break;
        case 3: /* Zephyr */
            if (tok_lookup(ps, "<|user|>", 8) >= 0) {
                if (!doe_snprintf_checked(wrapped, sizeof(wrapped), "Zephyr prompt",
                        "<|user|>\n%s\n<|assistant|>\n", input))
                    continue;
                use_template = 1;
            }
            break;
        case 4: /* Phi */
            if (tok_lookup(ps, "<|end|>", 7) >= 0) {
                if (!doe_snprintf_checked(wrapped, sizeof(wrapped), "Phi prompt",
                        "<|user|>\n%s<|end|>\n<|assistant|>\n", input))
                    continue;
                use_template = 1;
            }
            break;
        case 5: /* Gemma */
            if (tok_lookup(ps, "<start_of_turn>", 15) >= 0) {
                if (!doe_snprintf_checked(wrapped, sizeof(wrapped), "Gemma prompt",
                        "<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", input))
                    continue;
                use_template = 1;
            }
            break;
        case 6: /* nanollama — <|user_start|>...<|user_end|><|assistant_start|> */
            if (!doe_snprintf_checked(wrapped, sizeof(wrapped), "nanollama prompt",
                    "<|user_start|>%s<|user_end|><|assistant_start|>", input))
                continue;
            use_template = 1;
            break;
        }
        if (!use_template && !doe_snprintf_checked(wrapped, sizeof(wrapped), "raw prompt", "%s", input))
            continue;

        /* Tokenize wrapped input (buffer enlarged for vision: ~1022 image tokens) */
        int input_tokens[2048];
        int n_input = 0;
        infer_clear_image_embeds(&is);
        if (g_image_path[0] || g_img_embeds_bin[0]) {
            /* native pixtral vision — replicate mtmd layout:
             * <s> [INST] ▁ <img_count image embeds> [IMG_END] {prompt} ▁ [/INST] */
            int n_img = 0, vdim = ps->host_dim;
            float *emb = NULL;
            if (g_img_embeds_bin[0]) {
                /* diagnostic: load image embeds [n_tok * host_dim] f32 from file (clip ground-truth) */
                emb = load_img_embeds_bin(g_img_embeds_bin, ps->host_dim, &n_img);
                if (!emb) continue;
                printf("[doe] vision: %d image tokens from BIN %s\n", n_img, g_img_embeds_bin);
            } else {
                emb = pv_encode_image(g_image_path, g_mmproj_path, &n_img, &vdim);
                if (!emb) { printf("[doe] image encode failed: %s\n", g_image_path); continue; }
                printf("[doe] vision: %d image tokens (dim %d) from %s\n", n_img, vdim, g_image_path);
            }
            if (n_img <= 0 || vdim != ps->host_dim) {
                printf("[doe] vision embed shape mismatch: tokens=%d dim=%d host_dim=%d; refusing image splice\n",
                       n_img, vdim, ps->host_dim);
                free(emb);
                continue;
            }
            char vbuf[sizeof(input) + 2];
            if (!doe_snprintf_checked(vbuf, sizeof(vbuf), "vision prompt", "%s ", input)) {
                free(emb);
                continue;
            }
            int t;
            if (ps->bos_id >= 0 && !token_append_checked(ps->bos_id, input_tokens, 2048, &n_input, "vision BOS")) {
                free(emb);
                continue;
            }
            if ((t = tok_lookup(ps, "[INST]", 6)) >= 0 &&
                !token_append_checked(t, input_tokens, 2048, &n_input, "vision [INST]")) {
                free(emb);
                continue;
            }
            if (!tokenize_append_checked(ps, " ", input_tokens, 2048, &n_input, "vision prefix")) {
                free(emb);
                continue;
            }
            int img_room = 2000 - n_input;
            if (n_img < 0 || n_img > img_room) {
                printf("[doe] vision image tokens exceed prompt budget (%d > %d); refusing truncated image prompt\n",
                       n_img, img_room);
                free(emb);
                continue;
            }
            is.img_embeds = emb; is.img_start = n_input; is.img_count = n_img;
            int placeholder = ps->bos_id >= 0 ? ps->bos_id : 0;
            int image_splice_ok = 1;
            for (int k = 0; k < n_img; k++) {
                if (!token_append_checked(placeholder, input_tokens, 2048, &n_input, "vision image placeholder")) {
                    image_splice_ok = 0;
                    break;
                }
            }
            if (!image_splice_ok) {
                infer_clear_image_embeds(&is);
                continue;
            }
            if ((t = tok_lookup(ps, "[IMG_END]", 9)) >= 0 &&
                !token_append_checked(t, input_tokens, 2048, &n_input, "vision [IMG_END]")) {
                infer_clear_image_embeds(&is);
                continue;
            }
            if (!tokenize_append_checked(ps, vbuf, input_tokens, 2048, &n_input, "vision prompt")) {
                infer_clear_image_embeds(&is);
                continue;
            }
            if ((t = tok_lookup(ps, "[/INST]", 7)) >= 0 &&
                !token_append_checked(t, input_tokens, 2048, &n_input, "vision [/INST]")) {
                infer_clear_image_embeds(&is);
                continue;
            }
        } else {
            if (ps->bos_id >= 0 && !token_append_checked(ps->bos_id, input_tokens, 2048, &n_input, "chat BOS"))
                continue;
            if (!tokenize_append_checked(ps, wrapped, input_tokens, 2048, &n_input, "chat prompt"))
                continue;
        }
        if (n_input <= 0) {
            printf("[doe] prompt tokenization produced no tokens\n");
            infer_clear_image_embeds(&is);
            continue;
        }

        /* A12a: input-id dump WITH bos for parity vs llama (env-guarded, diagnostic only) */
        if (getenv("DOE_DEBUG_INPUT")) {
            fprintf(stderr, "[inputdump] n=%d ids:", n_input);
            for (int di = 0; di < n_input; di++) fprintf(stderr, " %d", input_tokens[di]);
            fprintf(stderr, "\n");
        }

        /* A14 (Mythos gen-only): repetition history = GENERATED tokens ONLY — prompt untouched.
         * A10a poisoned the start by penalizing prompt identity-tokens; here rep_hist stays empty
         * until generation begins. Window = last ~64 generated (applied below). */
        int rep_hist[256]; int n_rep = 0;

        int pos = 0;
        for (int i = 0; i < n_input - 1 && pos < max_seq - 1; i++, pos++) {
            doe_forward(ps, &is, input_tokens[i], pos);
            if (!(is.img_embeds && pos >= is.img_start && pos < is.img_start + is.img_count))
                dario_ingest(input_tokens[i]); /* feed user tokens into Dario field (skip image placeholders) */
        }

        int prev = input_tokens[n_input - 1];
        dario_ingest(prev); /* D-H2: last token still feeds the field; forwarded once in generation */
        printf("  ");
        int total_births = 0, total_deaths = 0;

        struct timespec _gt0, _gt1; clock_gettime(CLOCK_MONOTONIC, &_gt0); int _gi = 0;
        g_prof_mv_ns = 0; g_resident_ns = 0; g_head_ns = 0; /* reset: profile over decode loop only (exclude prefill) */
        for (int _g = 0; _g < MVG_N; _g++) { g_ps_ns[_g] = 0; g_ps_cnt[_g] = 0; }
        for (int i = 0; i < g_gen_max_new && pos < max_seq; i++, pos++) {
            _gi = i + 1;
            float *lg = doe_forward(ps, &is, prev, pos);

            /* A12b: raw host-forward token-1 NaN/Inf count + finite top-5 (before field/sample) */
            if (i == 0 && getenv("DOE_DEBUG_LOGITS")) {
                int V = ps->host_vocab, nnan = 0, ninf = 0;
                float *tmp = malloc(V * sizeof(float));
                if (!tmp) {
                    fprintf(stderr, "[logitdump] allocation failed for vocab=%d\n", V);
                    goto skip_logitdump;
                }
                for (int v = 0; v < V; v++) {
                    if (isnan(lg[v])) { nnan++; tmp[v] = -1e30f; }
                    else if (isinf(lg[v])) { ninf++; tmp[v] = -1e30f; }
                    else tmp[v] = lg[v];
                }
                fprintf(stderr, "[logitdump] tok1 nan=%d inf=%d /%d, finite top5:", nnan, ninf, V);
                for (int top = 0; top < 5; top++) {
                    int bi = 0; for (int v = 1; v < V; v++) if (tmp[v] > tmp[bi]) bi = v;
                    const char *ts = (ps->vocab_tokens && ps->vocab_tokens[bi]) ? ps->vocab_tokens[bi] : "?";
                    fprintf(stderr, " (%d '%s'=%.3f)", bi, ts, tmp[bi]);
                    tmp[bi] = -1e30f;
                }
                fprintf(stderr, "\n");
                free(tmp);
            }
skip_logitdump:

            /* Field modulation on logits — Dario Equation */
            field_step(1.0f);
            apply_field_to_logits(lg, ps->host_vocab);

            /* A10a: repetition penalty over recent history (llama-style), last 64 tokens */
            if (g_rep_penalty > 1.0f) {
                int rstart = n_rep > 64 ? n_rep - 64 : 0;
                for (int r = rstart; r < n_rep; r++) {
                    int t = rep_hist[r];
                    if (t >= 0 && t < ps->host_vocab)
                        lg[t] = lg[t] > 0 ? lg[t] / g_rep_penalty : lg[t] * g_rep_penalty;
                }
            }

            float sample_temp = g_gen_temp_override >= 0.0f ? g_gen_temp_override : F.effective_temp;
            int next = sample(lg, ps->host_vocab, sample_temp, g_gen_top_k);
            if (n_rep < 256) rep_hist[n_rep++] = next;

            /* Stop on EOS or chat-template end tokens */
            if (next == ps->eos_id) break;
            if (ps->vocab_tokens && next >= 0 && next < ps->vocab_size && ps->vocab_tokens[next]) {
                const char *ts = ps->vocab_tokens[next];
                if (strcmp(ts, "<|im_end|>") == 0 || strcmp(ts, "<|end|>") == 0 ||
                    strcmp(ts, "<|endoftext|>") == 0 || strcmp(ts, "<end_of_turn>") == 0 ||
                    strcmp(ts, "<|user|>") == 0 || strcmp(ts, "<|assistant_end|>") == 0 ||
                    strcmp(ts, "<|eot_id|>") == 0)
                    break;
            }

            /* Prophecy debt — retroactive conscience */
            float pd = compute_prophecy_debt(lg, next, ps->host_vocab);
            F.debt += pd;
            debt_sum += pd; debt_count++;

            /* Dario field: ingest generated token (co-occurrence + prophecy + destiny) */
            dario_ingest(next);

            /* NOTORCH Hebbian update — debt drives learning (A13b: gated by --train) */
            float learn_signal = pd > 0.3f ? -pd : (1.0f - pd) * 0.1f;
            if (g_train) for (int l = 0; l < ps->n_field_layers; l++) {
                FieldLayer *fl = &ps->field_layers[l];
                for (int e = 0; e < MAX_EXPERTS; e++) {
                    if (!fl->experts[e].alive || fl->experts[e].tokens_seen == 0) continue;
                    notorch_step(fl->experts[e].lora_A, fl->experts[e].lora_B,
                                ps->host_dim, ps->host_dim, ps->lora_rank,
                                is.x, is.xb, learn_signal);
                    /* A14: poison-quarantine (NaN/Inf + finite |value|-explosion) — freeze + death */
                    if (lora_poisoned(fl->experts[e].lora_A, fl->experts[e].lora_B,
                                      ps->host_dim * ps->lora_rank)) {
                        free_lora_expert(&fl->experts[e]);
                        field_recount_alive(fl);
                        total_deaths++;
                    }
                }
            }

            /* Vitality + mitosis + apoptosis */
            if (i % 10 == 0) {
                /* Harmonic decomposition */
                float lh[16]; int lhl = 0;
                for (int j = 0; j < 16 && j < i; j++) lh[lhl++] = F.entropy;
                if (lhl > 2) harmonic_decompose(&is.hs, lh, lhl);

                for (int l = 0; l < ps->n_field_layers; l++) {
                    update_expert_vitality(&ps->field_layers[l], 10);
                    if (try_mitosis(&ps->field_layers[l], ps->host_dim, ps->lora_rank)) total_births++;
                    if (try_apoptosis(&ps->field_layers[l])) total_deaths++;
                }
            }

            /* Drift snapshot */
            if (i % DRIFT_INTERVAL == 0 && i > 0)
                drift_snapshot(&cd, F.debt, ps, &is.hs);

            token_decode_print(ps, next);
            fflush(stdout);
            prev = next;
        }
        clock_gettime(CLOCK_MONOTONIC, &_gt1);
        if (getenv("DOE_DEBUG_TIMING") || g_prof_on > 0) {
            double _el = (_gt1.tv_sec - _gt0.tv_sec) + (_gt1.tv_nsec - _gt0.tv_nsec) / 1e9;
            fprintf(stderr, "[timing] decode %d tokens in %.2fs = %.2f t/s\n", _gi, _el, _el > 0 ? _gi / _el : 0.0);
            if (g_prof_on > 0) {
                double _mv = g_prof_mv_ns / 1e9;
                fprintf(stderr, "[profile] matvec %.2fs (%.0f%% of decode) | rest %.2fs (attention/rmsnorm/FFN/sample)\n",
                        _mv, _el > 0 ? 100 * _mv / _el : 0.0, _el - _mv);
                double _res = g_resident_ns/1e9, _hd = g_head_ns/1e9;
                fprintf(stderr, "[profile] resident-CB %.2fs (%.0f%%, %.1f ms/tok) | lm_head+norm %.2fs (%.0f%%, %.1f ms/tok) | other %.2fs\n",
                        _res, _el>0?100*_res/_el:0.0, _gi>0?_res*1e3/_gi:0.0,
                        _hd, _el>0?100*_hd/_el:0.0, _gi>0?_hd*1e3/_gi:0.0, _el-_res-_hd);
            }
            if (g_ps_on > 0 && _gi > 0) {
                static const char *_gn[MVG_N] = {"attn_qkv","attn_o","ffn_gate_up","ffn_down","lm_head"};
                double _tot = 0; for (int _g = 0; _g < MVG_N; _g++) _tot += g_ps_ns[_g];
                fprintf(stderr, "[per-shape] %d tokens, per-token (bytes-share check):\n", _gi);
                for (int _g = 0; _g < MVG_N; _g++)
                    fprintf(stderr, "  %-12s %.3f ms/tok  %.1f disp/tok  (m=%d k=%d dt=%d)  %.0f%% time\n",
                            _gn[_g], g_ps_ns[_g]/1e6/_gi, (double)g_ps_cnt[_g]/_gi,
                            g_ps_m[_g], g_ps_k[_g], g_ps_dt[_g], _tot > 0 ? 100*g_ps_ns[_g]/_tot : 0.0);
            }
        }
        printf("\n");

        /* Meta record */
        int te = 0;
        for (int l = 0; l < ps->n_field_layers; l++) te += ps->field_layers[l].n_alive;
        meta_record(&meta, F.step, te, ps->field_layers[0].parliament.consensus,
                    F.debt, F.field_health, debt_count > 0 ? debt_sum/debt_count : 0,
                    cd.drift, F.debt);

        if (total_births > 0 || total_deaths > 0)
            printf("  [life] births=%d deaths=%d\n", total_births, total_deaths);
        printf("\n");
        if (g_once) break;
    }
    free_infer(&is);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * HTTP SERVE MODE — minimal HTTP server for doe_ui.html, doe.html,
 * yent.html, worldmodel.html, and explicit worldmodel subassets.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static int g_serve_port = 0; /* 0 = disabled */
static int g_serve_public = 0; /* D-M4: 0 = bind 127.0.0.1 (default), 1 = 0.0.0.0 (--serve-public) */

/* JSON-escape a string into buf. Returns bytes written (not counting NUL), or -1 on truncation/error. */
static int json_escape(const char *src, char *buf, int bufsz) {
    if (!src || !buf || bufsz <= 0) return -1;
    int p = 0;
    for (; *src; src++) {
        const char *emit = src;
        int emit_len = 1;
        char esc[2];
        switch (*src) {
        case '"':  esc[0] = '\\'; esc[1] = '"';  emit = esc; emit_len = 2; break;
        case '\\': esc[0] = '\\'; esc[1] = '\\'; emit = esc; emit_len = 2; break;
        case '\n': esc[0] = '\\'; esc[1] = 'n';  emit = esc; emit_len = 2; break;
        case '\r': esc[0] = '\\'; esc[1] = 'r';  emit = esc; emit_len = 2; break;
        case '\t': esc[0] = '\\'; esc[1] = 't';  emit = esc; emit_len = 2; break;
        }
        if (p > bufsz - emit_len - 1) {
            buf[0] = '\0';
            return -1;
        }
        memcpy(buf + p, emit, (size_t)emit_len);
        p += emit_len;
    }
    buf[p] = '\0';
    return p;
}

static const char *http_header_value(const char *buf, int header_end, const char *name) {
    int name_len = (int)strlen(name);
    int line = 0;
    while (line < header_end) {
        int line_end = line;
        while (line_end + 1 < header_end && !(buf[line_end] == '\r' && buf[line_end + 1] == '\n'))
            line_end++;
        if (line_end > line + name_len && buf[line + name_len] == ':' &&
            strncasecmp(buf + line, name, (size_t)name_len) == 0) {
            const char *v = buf + line + name_len + 1;
            const char *end = buf + line_end;
            while (v < end && (*v == ' ' || *v == '\t')) v++;
            return v;
        }
        line = line_end + 2;
    }
    return NULL;
}

/* Read full HTTP request into buf, return total bytes. */
static int http_read_request(int fd, char *buf, int bufsz) {
    int total = 0;
    int content_length = -1;
    int header_end = -1;
    while (total < bufsz - 1) {
        int n = (int)read(fd, buf + total, bufsz - 1 - total);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        total += n;
        buf[total] = '\0';
        /* Find end of headers */
        if (header_end < 0) {
            char *hdr_end = strstr(buf, "\r\n\r\n");
            if (hdr_end) {
                header_end = (int)(hdr_end - buf) + 4;
                /* Parse Content-Length */
                const char *cl = http_header_value(buf, header_end, "content-length");
                if (cl) {
                    char *end = NULL;
                    errno = 0;
                    long v = strtol(cl, &end, 10);
                    while (end && (*end == ' ' || *end == '\t')) end++;
                    if (cl == end || errno || v < 0 || v > bufsz - header_end - 1 ||
                        !end || (*end != '\r' && *end != '\n')) {
                        fprintf(stderr, "[serve] invalid Content-Length\n");
                        return -1;
                    }
                    content_length = (int)v;
                } else {
                    content_length = 0;
                }
            }
        }
        if (header_end >= 0 && total >= header_end + content_length) break;
    }
    if (header_end < 0) {
        fprintf(stderr, "[serve] HTTP headers exceed request buffer\n");
        return -1;
    }
    if (total < header_end + content_length) {
        fprintf(stderr, "[serve] HTTP body truncated\n");
        return -1;
    }
    return total;
}

/* Send full buffer over socket */
static int http_send(int fd, const char *data, int len) {
    if (!data || len < 0) return 0;
    int sent = 0;
    while (sent < len) {
        int n = (int)write(fd, data + sent, len - sent);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return 0;
        sent += n;
    }
    return 1;
}

/* Send HTTP response header */
static int http_send_header(int fd, int status, const char *content_type, int content_length) {
    char hdr[512];
    const char *status_text = status == 200 ? "OK" :
                              status == 404 ? "Not Found" :
                              status == 500 ? "Internal Server Error" : "Bad Request";
    int hlen;
    if (content_length >= 0) {
        hlen = snprintf(hdr, sizeof(hdr),
            "HTTP/1.1 %d %s\r\n"
            "Content-Type: %s\r\n"
            "Content-Length: %d\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Headers: Content-Type\r\n"
            "Connection: close\r\n\r\n",
            status, status_text, content_type, content_length);
    } else {
        /* Streaming (SSE) — no content-length */
        hlen = snprintf(hdr, sizeof(hdr),
            "HTTP/1.1 %d %s\r\n"
            "Content-Type: %s\r\n"
            "Cache-Control: no-cache\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Headers: Content-Type\r\n"
            "Connection: keep-alive\r\n\r\n",
            status, status_text, content_type);
    }
    if (hlen < 0 || hlen >= (int)sizeof(hdr)) {
        fprintf(stderr, "[serve] HTTP response header too large\n");
        return 0;
    }
    return http_send(fd, hdr, hlen);
}

static void http_send_text(int fd, int status, const char *msg) {
    int len = msg ? (int)strlen(msg) : 0;
    if (!http_send_header(fd, status, "text/plain; charset=utf-8", len)) return;
    if (len > 0) http_send(fd, msg, len);
}

/* Serve a static file with a caller-selected content type. */
static int http_serve_static_file(int fd, const char *filepath, const char *content_type) {
    if (!content_type) content_type = "application/octet-stream";
    FILE *f = fopen(filepath, "rb");
    if (!f) return 0;
    if (fseek(f, 0, SEEK_END) != 0) {
        fprintf(stderr, "[serve] cannot seek static file: %s\n", filepath);
        fclose(f);
        http_send_text(fd, 500, "static file read failed");
        return 1;
    }
    long sz = ftell(f);
    if (sz < 0 || sz > INT_MAX || fseek(f, 0, SEEK_SET) != 0) {
        fprintf(stderr, "[serve] invalid static file size for %s\n", filepath);
        fclose(f);
        http_send_text(fd, 500, "static file read failed");
        return 1;
    }
    char *data = NULL;
    if (sz > 0) {
        data = malloc((size_t)sz);
        if (!data) {
            fprintf(stderr, "[serve] allocation failed for static file %s (%ld bytes)\n", filepath, sz);
            fclose(f);
            http_send_text(fd, 500, "static file read failed");
            return 1;
        }
        if (fread(data, 1, (size_t)sz, f) != (size_t)sz) {
            fprintf(stderr, "[serve] short read while serving static file: %s\n", filepath);
            free(data);
            fclose(f);
            http_send_text(fd, 500, "static file read failed");
            return 1;
        }
    }
    fclose(f);
    if (!http_send_header(fd, 200, content_type, (int)sz)) {
        free(data);
        return 1;
    }
    if (sz > 0) http_send(fd, data, (int)sz);
    free(data);
    return 1;
}

static int http_serve_file(int fd, const char *filepath) {
    return http_serve_static_file(fd, filepath, "text/html; charset=utf-8");
}

static int http_serve_first_file(int fd, const char *primary, const char *fallback, const char *content_type) {
    if (http_serve_static_file(fd, primary, content_type)) return 1;
    if (fallback && strcmp(primary, fallback) != 0)
        return http_serve_static_file(fd, fallback, content_type);
    return 0;
}

static float http_metric_float(float v, float fallback) {
    return isfinite(v) ? v : fallback;
}

static int http_alive_expert_count(const GGUFIndex *ps) {
    if (!ps || ps->n_field_layers <= 0) return 0;
    int total = 0;
    for (int l = 0; l < ps->n_field_layers; l++)
        if (ps->field_layers[l].n_alive > 0) total += ps->field_layers[l].n_alive;
    return total;
}

static float http_layer0_consensus(const GGUFIndex *ps) {
    if (!ps || ps->n_field_layers <= 0) return 0.0f;
    float c = ps->field_layers[0].parliament.consensus;
    return isfinite(c) ? clamp01(c) : 0.5f;
}

typedef struct {
    int id;
    float prob;
} HttpTopToken;

static float http_logprob_from_prob(float p) {
    if (!isfinite(p) || p <= 0.0f) p = 1e-30f;
    return logf(p);
}

static void http_collect_top_tokens(const float *probs, int vocab, int selected,
                                    HttpTopToken *top, int top_cap, int *top_n,
                                    float *tail_mass, float *selected_prob,
                                    int *selected_rank) {
    if (top_n) *top_n = 0;
    if (tail_mass) *tail_mass = 0.0f;
    if (selected_prob) *selected_prob = 0.0f;
    if (selected_rank) *selected_rank = 0;
    if (!probs || vocab <= 0 || !top || top_cap <= 0) return;

    float total = 0.0f;
    float sp = 0.0f;
    if (selected >= 0 && selected < vocab && isfinite(probs[selected]) && probs[selected] > 0.0f)
        sp = probs[selected];
    int rank = sp > 0.0f ? 1 : 0;
    int n = 0;
    for (int i = 0; i < vocab; i++) {
        float p = probs[i];
        if (!isfinite(p) || p <= 0.0f) continue;
        total += p;
        if (sp > 0.0f && p > sp) rank++;
        int at = n;
        if (n < top_cap) {
            n++;
        } else if (p > top[n - 1].prob) {
            at = n - 1;
        } else {
            continue;
        }
        while (at > 0 && p > top[at - 1].prob) {
            top[at] = top[at - 1];
            at--;
        }
        top[at].id = i;
        top[at].prob = p;
    }

    float shown = 0.0f;
    for (int i = 0; i < n; i++) shown += top[i].prob;
    if (top_n) *top_n = n;
    if (tail_mass) *tail_mass = clamp01(total - shown);
    if (selected_prob) *selected_prob = sp;
    if (selected_rank) *selected_rank = rank;
}

static int http_appendf(char *buf, int bufsz, int *used, const char *fmt, ...) {
    if (!buf || !used || *used < 0 || *used >= bufsz) return 0;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf + *used, (size_t)(bufsz - *used), fmt, ap);
    va_end(ap);
    if (n < 0 || n >= bufsz - *used) return 0;
    *used += n;
    return 1;
}

static int http_top_tokens_json(GGUFIndex *ps, const HttpTopToken *top, int top_n,
                                int selected, char *buf, int bufsz) {
    if (!buf || bufsz <= 0) return 0;
    int used = 0;
    if (!http_appendf(buf, bufsz, &used, "[")) return 0;
    for (int i = 0; i < top_n; i++) {
        char tok[64], esc[160];
        token_decode_buf(ps, top[i].id, tok, sizeof(tok));
        if (json_escape(tok, esc, sizeof(esc)) < 0) esc[0] = '\0';
        if (!http_appendf(buf, bufsz, &used,
                          "%s{\"id\":%d,\"token\":\"%s\",\"prob\":%.6g,"
                          "\"logprob\":%.6g,\"selected\":%s}",
                          i ? "," : "", top[i].id, esc,
                          http_metric_float(top[i].prob, 0.0f),
                          http_logprob_from_prob(top[i].prob),
                          top[i].id == selected ? "true" : "false"))
            return 0;
    }
    return http_appendf(buf, bufsz, &used, "]");
}

/* Extract JSON string value for a key from body. Simple parser. */
static int json_get_string(const char *json, const char *key, char *out, int outsz) {
    if (!json || !key || !out || outsz <= 0) return -1;
    out[0] = '\0';
    char needle[64];
    int nneedle = snprintf(needle, sizeof(needle), "\"%s\"", key);
    if (nneedle < 0 || nneedle >= (int)sizeof(needle)) return -1;
    const char *p = strstr(json, needle);
    if (!p) return 0;
    p = strchr(p + strlen(needle), ':');
    if (!p) return -1;
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (*p != '"') return -1;
    p++;
    int i = 0;
    while (*p && *p != '"' && i < outsz - 1) {
        if (*p == '\\' && p[1]) { p++; /* skip escape */ }
        out[i++] = *p++;
    }
    out[i] = '\0';
    if (*p != '"') return -1;
    return i;
}

/* Extract last user message from messages array in chat/completions body */
static int json_get_last_user_message(const char *body, char *out, int outsz) {
    /* Find last "role":"user" ... "content":"..." */
    const char *last_user = NULL;
    const char *p = body;
    while ((p = strstr(p, "\"role\"")) != NULL) {
        const char *rv = strstr(p, "\"user\"");
        if (rv && rv - p < 30) last_user = p;
        p++;
    }
    if (!last_user) return 0;
    return json_get_string(last_user, "content", out, outsz);
}

static int json_get_number(const char *json, const char *key, float *out) {
    char needle[64];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = strstr(json, needle);
    if (!p) return 0;
    p = strchr(p + strlen(needle), ':');
    if (!p) return 0;
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    errno = 0;
    char *end = NULL;
    float v = strtof(p, &end);
    if (p == end || errno || !isfinite(v)) return 0;
    while (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r') end++;
    if (*end && *end != ',' && *end != '}' && *end != ']') return 0;
    if (out) *out = v;
    return 1;
}

static float json_get_float(const char *json, const char *key, float def) {
    float v = def;
    if (!json_get_number(json, key, &v)) return def;
    return v;
}

/* Run inference and stream SSE tokens */
static void http_stream_inference(int fd, GGUFIndex *ps, const char *user_msg, float temperature, int max_tokens) {
    int max_seq = 512;
    InferState is = alloc_infer(ps, max_seq);
    if (!is.x) {
        const char *err = "data: {\"error\":\"inference allocation failed\"}\n\n";
        http_send(fd, err, (int)strlen(err));
        return;
    }

    /* Reset KV cache */
    memset(is.key_cache, 0, is.kv_bytes);
    memset(is.value_cache, 0, is.kv_bytes);

    /* Wrap input in chat template */
    char wrapped[4096];
    int wrapped_ok = 0;
    switch (ps->chat_style) {
    case 1:
        wrapped_ok = doe_snprintf_checked(wrapped, sizeof(wrapped), "HTTP ChatML prompt",
                                          "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user_msg);
        break;
    case 2:
        wrapped_ok = doe_snprintf_checked(wrapped, sizeof(wrapped), "HTTP INST prompt",
                                          "[INST] %s [/INST]", user_msg);
        break;
    case 3:
        wrapped_ok = doe_snprintf_checked(wrapped, sizeof(wrapped), "HTTP Zephyr prompt",
                                          "<|user|>\n%s\n<|assistant|>\n", user_msg);
        break;
    case 4:
        wrapped_ok = doe_snprintf_checked(wrapped, sizeof(wrapped), "HTTP Phi prompt",
                                          "<|user|>\n%s<|end|>\n<|assistant|>\n", user_msg);
        break;
    case 5:
        wrapped_ok = doe_snprintf_checked(wrapped, sizeof(wrapped), "HTTP Gemma prompt",
                                          "<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", user_msg);
        break;
    case 6:
        wrapped_ok = doe_snprintf_checked(wrapped, sizeof(wrapped), "HTTP nanollama prompt",
                                          "<|user_start|>%s<|user_end|><|assistant_start|>", user_msg);
        break;
    default:
        wrapped_ok = doe_snprintf_checked(wrapped, sizeof(wrapped), "HTTP raw prompt", "%s", user_msg);
        break;
    }
    if (!wrapped_ok) {
        const char *err = "data: {\"error\":\"prompt too long\"}\n\n";
        http_send(fd, err, (int)strlen(err));
        free_infer(&is);
        return;
    }

    /* Tokenize */
    int input_tokens[512];
    int n_input = 0;
    if (ps->bos_id >= 0 && !token_append_checked(ps->bos_id, input_tokens, 512, &n_input, "HTTP BOS")) {
        const char *err = "data: {\"error\":\"prompt token budget exceeded\"}\n\n";
        http_send(fd, err, (int)strlen(err));
        free_infer(&is);
        return;
    }
    if (!tokenize_append_checked(ps, wrapped, input_tokens, 512, &n_input, "HTTP prompt")) {
        const char *err = "data: {\"error\":\"prompt token budget exceeded\"}\n\n";
        http_send(fd, err, (int)strlen(err));
        free_infer(&is);
        return;
    }
    if (n_input <= 0) {
        const char *err = "data: {\"error\":\"prompt tokenization failed\"}\n\n";
        http_send(fd, err, (int)strlen(err));
        free_infer(&is);
        return;
    }

    /* Prefill */
    int pos = 0;
    for (int i = 0; i < n_input - 1 && pos < max_seq - 1; i++, pos++) {
        doe_forward(ps, &is, input_tokens[i], pos);
        dario_ingest(input_tokens[i]); /* feed user tokens into Dario field */
    }

    int prev = input_tokens[n_input - 1];
    dario_ingest(prev); /* D-H2: last token still feeds the field; forwarded once in generation */

    /* Generate tokens, stream as SSE */
    for (int i = 0; i < max_tokens && pos < max_seq; i++, pos++) {
        float *lg = doe_forward(ps, &is, prev, pos);
        field_step(1.0f);
        apply_field_to_logits(lg, ps->host_vocab);

        float temp = temperature > 0.01f ? temperature : F.effective_temp;
        int next = sample(lg, ps->host_vocab, temp, 40);
        HttpTopToken top[HTTP_SSE_TOP_N];
        int n_top = 0, selected_rank = 0;
        float tail_mass = 0.0f, selected_prob = 0.0f;
        http_collect_top_tokens(lg, ps->host_vocab, next, top, HTTP_SSE_TOP_N, &n_top,
                                &tail_mass, &selected_prob, &selected_rank);

        /* Stop on EOS */
        if (next == ps->eos_id) break;
        if (ps->vocab_tokens && next >= 0 && next < ps->vocab_size && ps->vocab_tokens[next]) {
            const char *ts = ps->vocab_tokens[next];
            if (strcmp(ts, "<|im_end|>") == 0 || strcmp(ts, "<|end|>") == 0 ||
                strcmp(ts, "<|endoftext|>") == 0 || strcmp(ts, "<end_of_turn>") == 0 ||
                strcmp(ts, "<|user|>") == 0 || strcmp(ts, "<|assistant_end|>") == 0 ||
                strcmp(ts, "<|eot_id|>") == 0) break;
        }

        /* Prophecy debt + Hebbian update */
        float pd = compute_prophecy_debt(lg, next, ps->host_vocab);
        F.debt += pd;

        /* Dario field: ingest generated token */
        dario_ingest(next);

        float learn_signal = pd > 0.3f ? -pd : (1.0f - pd) * 0.1f;
        if (g_train) for (int l = 0; l < ps->n_field_layers; l++) {
            FieldLayer *fl = &ps->field_layers[l];
            for (int e = 0; e < MAX_EXPERTS; e++) {
                if (!fl->experts[e].alive || fl->experts[e].tokens_seen == 0) continue;
                notorch_step(fl->experts[e].lora_A, fl->experts[e].lora_B,
                            ps->host_dim, ps->host_dim, ps->lora_rank,
                            is.x, is.xb, learn_signal);
                /* A14: poison-quarantine (NaN/Inf + finite |value|-explosion) — freeze + death */
                if (lora_poisoned(fl->experts[e].lora_A, fl->experts[e].lora_B,
                                  ps->host_dim * ps->lora_rank)) {
                    free_lora_expert(&fl->experts[e]);
                    field_recount_alive(fl);
                }
            }
        }

        /* Decode token to buffer */
        char tokbuf[256], escaped[512];
        token_decode_buf(ps, next, tokbuf, sizeof(tokbuf));
        if (json_escape(tokbuf, escaped, sizeof(escaped)) < 0) {
            const char *err = "data: {\"error\":\"token escape overflow\"}\n\n";
            http_send(fd, err, (int)strlen(err));
            break;
        }

        /* Send SSE event with real runtime telemetry for UI observers. */
        char top_json[2048];
        if (!http_top_tokens_json(ps, top, n_top, next, top_json, sizeof(top_json)))
            strcpy(top_json, "[]");
        char sse[4096];
        int slen = snprintf(sse, sizeof(sse),
            "data: {\"token\":\"%s\",\"token_id\":%d,\"step\":%d,"
            "\"experts\":%d,\"debt\":%.6g,\"prophecy_debt\":%.6g,"
            "\"field_health\":%.6g,\"consensus\":%.6g,"
            "\"entropy\":%.6g,\"resonance\":%.6g,\"emergence\":%.6g,"
            "\"temperature\":%.6g,\"selected_prob\":%.6g,"
            "\"selected_logprob\":%.6g,\"selected_rank\":%d,"
            "\"candidate_tail_mass\":%.6g,\"top_tokens\":%s}\n\n",
            escaped, next, F.step,
            http_alive_expert_count(ps),
            http_metric_float(F.debt, 0.0f),
            http_metric_float(pd, 0.0f),
            http_metric_float(F.field_health, 0.0f),
            http_layer0_consensus(ps),
            http_metric_float(F.entropy, 0.0f),
            http_metric_float(F.resonance, 0.0f),
            http_metric_float(F.emergence, 0.0f),
            http_metric_float(temp, 0.0f),
            http_metric_float(selected_prob, 0.0f),
            http_logprob_from_prob(selected_prob),
            selected_rank,
            http_metric_float(tail_mass, 0.0f),
            top_json);
        if (slen < 0 || slen >= (int)sizeof(sse)) {
            const char *err = "data: {\"error\":\"token event too large\"}\n\n";
            http_send(fd, err, (int)strlen(err));
            break;
        }
        if (!http_send(fd, sse, slen)) break; /* client disconnected */

        prev = next;
    }

    /* Send done event */
    static const char done_event[] = "data: {\"done\":true}\n\n";
    http_send(fd, done_event, (int)sizeof(done_event) - 1); /* D-L2: full event incl. both \n */
    free_infer(&is);
}

/* Main HTTP serve loop */
static void serve_loop(GGUFIndex *ps, const char *exe_dir) {
    signal(SIGPIPE, SIG_IGN); /* ignore broken pipes */

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("[serve] socket"); return; }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = g_serve_public ? INADDR_ANY : htonl(INADDR_LOOPBACK); /* D-M4: localhost by default */
    addr.sin_port = htons(g_serve_port);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("[serve] bind"); close(server_fd); return;
    }
    if (listen(server_fd, 8) < 0) {
        perror("[serve] listen"); close(server_fd); return;
    }

    /* Resolve HTML file paths relative to executable */
    char ui_path[512], vis_path[512], yent_path[512], worldmodel_path[512];
    char yent_adj_path[512], worldmodel_adj_path[512];
    char interface_session_js_path[512], interface_restore_js_path[512], event_stream_js_path[512], chat_stream_js_path[512], interface_text_js_path[512], token_telemetry_js_path[512], interface_state_js_path[512], interface_clock_js_path[512], interface_status_js_path[512], interface_output_js_path[512], interface_transcript_js_path[512], interface_hud_js_path[512], interface_replay_js_path[512], interface_input_js_path[512], interface_events_js_path[512], interface_turn_js_path[512], interface_submit_js_path[512], interface_outcome_js_path[512], interface_run_js_path[512], interface_boot_js_path[512], interface_animation_js_path[512], interface_math_js_path[512], interface_canvas_js_path[512], interface_style_js_path[512], interface_deps_js_path[512];
    char yent_js_path[512], worldmodel_geometry_js_path[512], worldmodel_js_path[512];
    int ui_len = snprintf(ui_path, sizeof(ui_path), "%sdoe_ui.html", exe_dir);
    int vis_len = snprintf(vis_path, sizeof(vis_path), "%sdoe.html", exe_dir);
    int yent_len = snprintf(yent_path, sizeof(yent_path), "%s../yent.html", exe_dir);
    int worldmodel_len = snprintf(worldmodel_path, sizeof(worldmodel_path), "%s../worldmodel.html", exe_dir);
    int yent_adj_len = snprintf(yent_adj_path, sizeof(yent_adj_path), "%syent.html", exe_dir);
    int worldmodel_adj_len = snprintf(worldmodel_adj_path, sizeof(worldmodel_adj_path), "%sworldmodel.html", exe_dir);
    int interface_session_js_len = snprintf(interface_session_js_path, sizeof(interface_session_js_path), "%sworldmodel/interface_session.js", exe_dir);
    int interface_restore_js_len = snprintf(interface_restore_js_path, sizeof(interface_restore_js_path), "%sworldmodel/interface_restore.js", exe_dir);
    int event_stream_js_len = snprintf(event_stream_js_path, sizeof(event_stream_js_path), "%sworldmodel/event_stream.js", exe_dir);
    int chat_stream_js_len = snprintf(chat_stream_js_path, sizeof(chat_stream_js_path), "%sworldmodel/chat_stream.js", exe_dir);
    int interface_text_js_len = snprintf(interface_text_js_path, sizeof(interface_text_js_path), "%sworldmodel/interface_text.js", exe_dir);
    int token_telemetry_js_len = snprintf(token_telemetry_js_path, sizeof(token_telemetry_js_path), "%sworldmodel/token_telemetry.js", exe_dir);
    int interface_state_js_len = snprintf(interface_state_js_path, sizeof(interface_state_js_path), "%sworldmodel/interface_state.js", exe_dir);
    int interface_clock_js_len = snprintf(interface_clock_js_path, sizeof(interface_clock_js_path), "%sworldmodel/interface_clock.js", exe_dir);
    int interface_status_js_len = snprintf(interface_status_js_path, sizeof(interface_status_js_path), "%sworldmodel/interface_status.js", exe_dir);
    int interface_output_js_len = snprintf(interface_output_js_path, sizeof(interface_output_js_path), "%sworldmodel/interface_output.js", exe_dir);
    int interface_transcript_js_len = snprintf(interface_transcript_js_path, sizeof(interface_transcript_js_path), "%sworldmodel/interface_transcript.js", exe_dir);
    int interface_hud_js_len = snprintf(interface_hud_js_path, sizeof(interface_hud_js_path), "%sworldmodel/interface_hud.js", exe_dir);
    int interface_replay_js_len = snprintf(interface_replay_js_path, sizeof(interface_replay_js_path), "%sworldmodel/interface_replay.js", exe_dir);
    int interface_input_js_len = snprintf(interface_input_js_path, sizeof(interface_input_js_path), "%sworldmodel/interface_input.js", exe_dir);
    int interface_events_js_len = snprintf(interface_events_js_path, sizeof(interface_events_js_path), "%sworldmodel/interface_events.js", exe_dir);
    int interface_turn_js_len = snprintf(interface_turn_js_path, sizeof(interface_turn_js_path), "%sworldmodel/interface_turn.js", exe_dir);
    int interface_submit_js_len = snprintf(interface_submit_js_path, sizeof(interface_submit_js_path), "%sworldmodel/interface_submit.js", exe_dir);
    int interface_outcome_js_len = snprintf(interface_outcome_js_path, sizeof(interface_outcome_js_path), "%sworldmodel/interface_outcome.js", exe_dir);
    int interface_run_js_len = snprintf(interface_run_js_path, sizeof(interface_run_js_path), "%sworldmodel/interface_run.js", exe_dir);
    int interface_boot_js_len = snprintf(interface_boot_js_path, sizeof(interface_boot_js_path), "%sworldmodel/interface_boot.js", exe_dir);
    int interface_animation_js_len = snprintf(interface_animation_js_path, sizeof(interface_animation_js_path), "%sworldmodel/interface_animation.js", exe_dir);
    int interface_math_js_len = snprintf(interface_math_js_path, sizeof(interface_math_js_path), "%sworldmodel/interface_math.js", exe_dir);
    int interface_canvas_js_len = snprintf(interface_canvas_js_path, sizeof(interface_canvas_js_path), "%sworldmodel/interface_canvas.js", exe_dir);
    int interface_style_js_len = snprintf(interface_style_js_path, sizeof(interface_style_js_path), "%sworldmodel/interface_style.js", exe_dir);
    int interface_deps_js_len = snprintf(interface_deps_js_path, sizeof(interface_deps_js_path), "%sworldmodel/interface_deps.js", exe_dir);
    int yent_js_len = snprintf(yent_js_path, sizeof(yent_js_path), "%sworldmodel/yent.js", exe_dir);
    int worldmodel_geometry_js_len = snprintf(worldmodel_geometry_js_path, sizeof(worldmodel_geometry_js_path), "%sworldmodel/worldmodel_geometry.js", exe_dir);
    int worldmodel_js_len = snprintf(worldmodel_js_path, sizeof(worldmodel_js_path), "%sworldmodel/worldmodel.js", exe_dir);
    if (ui_len < 0 || ui_len >= (int)sizeof(ui_path) ||
        vis_len < 0 || vis_len >= (int)sizeof(vis_path) ||
        yent_len < 0 || yent_len >= (int)sizeof(yent_path) ||
        worldmodel_len < 0 || worldmodel_len >= (int)sizeof(worldmodel_path) ||
        yent_adj_len < 0 || yent_adj_len >= (int)sizeof(yent_adj_path) ||
        worldmodel_adj_len < 0 || worldmodel_adj_len >= (int)sizeof(worldmodel_adj_path) ||
        interface_session_js_len < 0 || interface_session_js_len >= (int)sizeof(interface_session_js_path) ||
        interface_restore_js_len < 0 || interface_restore_js_len >= (int)sizeof(interface_restore_js_path) ||
        event_stream_js_len < 0 || event_stream_js_len >= (int)sizeof(event_stream_js_path) ||
        chat_stream_js_len < 0 || chat_stream_js_len >= (int)sizeof(chat_stream_js_path) ||
        interface_text_js_len < 0 || interface_text_js_len >= (int)sizeof(interface_text_js_path) ||
        token_telemetry_js_len < 0 || token_telemetry_js_len >= (int)sizeof(token_telemetry_js_path) ||
        interface_state_js_len < 0 || interface_state_js_len >= (int)sizeof(interface_state_js_path) ||
        interface_clock_js_len < 0 || interface_clock_js_len >= (int)sizeof(interface_clock_js_path) ||
        interface_status_js_len < 0 || interface_status_js_len >= (int)sizeof(interface_status_js_path) ||
        interface_output_js_len < 0 || interface_output_js_len >= (int)sizeof(interface_output_js_path) ||
        interface_transcript_js_len < 0 || interface_transcript_js_len >= (int)sizeof(interface_transcript_js_path) ||
        interface_hud_js_len < 0 || interface_hud_js_len >= (int)sizeof(interface_hud_js_path) ||
        interface_replay_js_len < 0 || interface_replay_js_len >= (int)sizeof(interface_replay_js_path) ||
        interface_input_js_len < 0 || interface_input_js_len >= (int)sizeof(interface_input_js_path) ||
        interface_events_js_len < 0 || interface_events_js_len >= (int)sizeof(interface_events_js_path) ||
        interface_turn_js_len < 0 || interface_turn_js_len >= (int)sizeof(interface_turn_js_path) ||
        interface_submit_js_len < 0 || interface_submit_js_len >= (int)sizeof(interface_submit_js_path) ||
        interface_outcome_js_len < 0 || interface_outcome_js_len >= (int)sizeof(interface_outcome_js_path) ||
        interface_run_js_len < 0 || interface_run_js_len >= (int)sizeof(interface_run_js_path) ||
        interface_boot_js_len < 0 || interface_boot_js_len >= (int)sizeof(interface_boot_js_path) ||
        interface_animation_js_len < 0 || interface_animation_js_len >= (int)sizeof(interface_animation_js_path) ||
        interface_math_js_len < 0 || interface_math_js_len >= (int)sizeof(interface_math_js_path) ||
        interface_canvas_js_len < 0 || interface_canvas_js_len >= (int)sizeof(interface_canvas_js_path) ||
        interface_style_js_len < 0 || interface_style_js_len >= (int)sizeof(interface_style_js_path) ||
        interface_deps_js_len < 0 || interface_deps_js_len >= (int)sizeof(interface_deps_js_path) ||
        yent_js_len < 0 || yent_js_len >= (int)sizeof(yent_js_path) ||
        worldmodel_geometry_js_len < 0 || worldmodel_geometry_js_len >= (int)sizeof(worldmodel_geometry_js_path) ||
        worldmodel_js_len < 0 || worldmodel_js_len >= (int)sizeof(worldmodel_js_path)) {
        fprintf(stderr, "[serve] static file path too long\n");
        close(server_fd);
        return;
    }

    printf("[serve] parliament listening on http://%s:%d\n", g_serve_public ? "0.0.0.0" : "127.0.0.1", g_serve_port);
    printf("[serve]   /         → chat UI\n");
    printf("[serve]   /visual   → parliament terminal\n");
    printf("[serve]   /yent     → Janus parliament interface\n");
    printf("[serve]   /worldmodel → walkable probability field\n");
    printf("[serve]   /health   → status\n");
    printf("[serve]   POST /chat/completions → inference stream\n\n");

    while (1) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client < 0) continue;

        char req[8192];
        int reqlen = http_read_request(client, req, sizeof(req));
        if (reqlen < 0) {
            http_send_text(client, 400, "bad request");
            close(client);
            continue;
        }
        if (reqlen == 0) { close(client); continue; }

        /* Parse method and path */
        char method[8] = "", path[256] = "";
        if (sscanf(req, "%7s %255s", method, path) != 2) {
            http_send_text(client, 400, "bad request");
            close(client);
            continue;
        }
        char *query = strchr(path, '?');
        if (query) *query = '\0';

        /* Handle CORS preflight */
        if (strcmp(method, "OPTIONS") == 0) {
            const char *cors = "HTTP/1.1 204 No Content\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                "Access-Control-Allow-Headers: Content-Type\r\n"
                "Content-Length: 0\r\n"
                "Connection: close\r\n\r\n";
            http_send(client, cors, (int)strlen(cors));
            close(client);
            continue;
        }

        if (strcmp(method, "GET") == 0) {
            if (strcmp(path, "/") == 0 || strcmp(path, "/index.html") == 0) {
                if (!http_serve_file(client, ui_path)) {
                    const char *msg = "doe_ui.html not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/visual") == 0) {
                if (!http_serve_file(client, vis_path)) {
                    const char *msg = "doe.html not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/yent") == 0 || strcmp(path, "/yent.html") == 0) {
                if (!http_serve_first_file(client, yent_path, yent_adj_path, "text/html; charset=utf-8")) {
                    const char *msg = "yent.html not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/worldmodel") == 0 || strcmp(path, "/worldmodel.html") == 0) {
                if (!http_serve_first_file(client, worldmodel_path, worldmodel_adj_path, "text/html; charset=utf-8")) {
                    const char *msg = "worldmodel.html not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/worldmodel/interface_session.js") == 0) {
                if (!http_serve_static_file(client, interface_session_js_path, "application/javascript; charset=utf-8")) {
                    const char *msg = "worldmodel/interface_session.js not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/worldmodel/interface_restore.js") == 0) {
                if (!http_serve_static_file(client, interface_restore_js_path, "application/javascript; charset=utf-8")) {
                    const char *msg = "worldmodel/interface_restore.js not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/worldmodel/event_stream.js") == 0) {
                if (!http_serve_static_file(client, event_stream_js_path, "application/javascript; charset=utf-8")) {
                    const char *msg = "worldmodel/event_stream.js not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/worldmodel/chat_stream.js") == 0) {
                if (!http_serve_static_file(client, chat_stream_js_path, "application/javascript; charset=utf-8")) {
                    const char *msg = "worldmodel/chat_stream.js not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/worldmodel/interface_text.js") == 0) {
                if (!http_serve_static_file(client, interface_text_js_path, "application/javascript; charset=utf-8")) {
                    const char *msg = "worldmodel/interface_text.js not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/worldmodel/token_telemetry.js") == 0) {
                if (!http_serve_static_file(client, token_telemetry_js_path, "application/javascript; charset=utf-8")) {
                    const char *msg = "worldmodel/token_telemetry.js not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/worldmodel/interface_state.js") == 0) {
                if (!http_serve_static_file(client, interface_state_js_path, "application/javascript; charset=utf-8")) {
                    const char *msg = "worldmodel/interface_state.js not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/worldmodel/interface_clock.js") == 0) {
                if (!http_serve_static_file(client, interface_clock_js_path, "application/javascript; charset=utf-8")) {
                    const char *msg = "worldmodel/interface_clock.js not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/worldmodel/interface_status.js") == 0) {
                if (!http_serve_static_file(client, interface_status_js_path, "application/javascript; charset=utf-8")) {
                    const char *msg = "worldmodel/interface_status.js not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/worldmodel/interface_output.js") == 0) {
                if (!http_serve_static_file(client, interface_output_js_path, "application/javascript; charset=utf-8")) {
                    const char *msg = "worldmodel/interface_output.js not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/worldmodel/interface_transcript.js") == 0) {
                if (!http_serve_static_file(client, interface_transcript_js_path, "application/javascript; charset=utf-8")) {
                    const char *msg = "worldmodel/interface_transcript.js not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/worldmodel/interface_hud.js") == 0) {
                if (!http_serve_static_file(client, interface_hud_js_path, "application/javascript; charset=utf-8")) {
                    const char *msg = "worldmodel/interface_hud.js not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/worldmodel/interface_replay.js") == 0) {
                if (!http_serve_static_file(client, interface_replay_js_path, "application/javascript; charset=utf-8")) {
                    const char *msg = "worldmodel/interface_replay.js not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/worldmodel/interface_input.js") == 0) {
                if (!http_serve_static_file(client, interface_input_js_path, "application/javascript; charset=utf-8")) {
                    const char *msg = "worldmodel/interface_input.js not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/worldmodel/interface_events.js") == 0) {
                if (!http_serve_static_file(client, interface_events_js_path, "application/javascript; charset=utf-8")) {
                    const char *msg = "worldmodel/interface_events.js not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/worldmodel/interface_turn.js") == 0) {
                if (!http_serve_static_file(client, interface_turn_js_path, "application/javascript; charset=utf-8")) {
                    const char *msg = "worldmodel/interface_turn.js not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/worldmodel/interface_submit.js") == 0) {
                if (!http_serve_static_file(client, interface_submit_js_path, "application/javascript; charset=utf-8")) {
                    const char *msg = "worldmodel/interface_submit.js not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/worldmodel/interface_outcome.js") == 0) {
                if (!http_serve_static_file(client, interface_outcome_js_path, "application/javascript; charset=utf-8")) {
                    const char *msg = "worldmodel/interface_outcome.js not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/worldmodel/interface_run.js") == 0) {
                if (!http_serve_static_file(client, interface_run_js_path, "application/javascript; charset=utf-8")) {
                    const char *msg = "worldmodel/interface_run.js not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/worldmodel/interface_boot.js") == 0) {
                if (!http_serve_static_file(client, interface_boot_js_path, "application/javascript; charset=utf-8")) {
                    const char *msg = "worldmodel/interface_boot.js not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/worldmodel/interface_animation.js") == 0) {
                if (!http_serve_static_file(client, interface_animation_js_path, "application/javascript; charset=utf-8")) {
                    const char *msg = "worldmodel/interface_animation.js not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/worldmodel/interface_math.js") == 0) {
                if (!http_serve_static_file(client, interface_math_js_path, "application/javascript; charset=utf-8")) {
                    const char *msg = "worldmodel/interface_math.js not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/worldmodel/interface_canvas.js") == 0) {
                if (!http_serve_static_file(client, interface_canvas_js_path, "application/javascript; charset=utf-8")) {
                    const char *msg = "worldmodel/interface_canvas.js not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/worldmodel/interface_style.js") == 0) {
                if (!http_serve_static_file(client, interface_style_js_path, "application/javascript; charset=utf-8")) {
                    const char *msg = "worldmodel/interface_style.js not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/worldmodel/interface_deps.js") == 0) {
                if (!http_serve_static_file(client, interface_deps_js_path, "application/javascript; charset=utf-8")) {
                    const char *msg = "worldmodel/interface_deps.js not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/worldmodel/yent.js") == 0) {
                if (!http_serve_static_file(client, yent_js_path, "application/javascript; charset=utf-8")) {
                    const char *msg = "worldmodel/yent.js not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/worldmodel/worldmodel_geometry.js") == 0) {
                if (!http_serve_static_file(client, worldmodel_geometry_js_path, "application/javascript; charset=utf-8")) {
                    const char *msg = "worldmodel/worldmodel_geometry.js not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/worldmodel/worldmodel.js") == 0) {
                if (!http_serve_static_file(client, worldmodel_js_path, "application/javascript; charset=utf-8")) {
                    const char *msg = "worldmodel/worldmodel.js not found";
                    http_send_text(client, 404, msg);
                }
            } else if (strcmp(path, "/health") == 0) {
                char body[512];
                char host_path_json[512], host_arch_json[128];
                if (json_escape(ps->host_path, host_path_json, sizeof(host_path_json)) < 0 ||
                    json_escape(ps->host_arch, host_arch_json, sizeof(host_arch_json)) < 0) {
                    http_send_text(client, 500, "health response too large");
                    close(client);
                    continue;
                }
                int blen = snprintf(body, sizeof(body),
                    "{\"status\":\"ok\",\"model\":\"%s\",\"arch\":\"%s\","
                    "\"params\":\"%dM\",\"vocab\":%d,\"layers\":%d,"
                    "\"experts\":%d,\"debt\":%.4f,\"health\":%.4f}",
                    host_path_json, host_arch_json,
                    (int)(ps->host_n_params / 1000000),
                    ps->host_vocab, ps->host_n_layers,
                    ps->n_field_layers > 0 ? ps->field_layers[0].n_alive : 0,
                    F.debt, F.field_health);
                if (blen < 0 || blen >= (int)sizeof(body)) {
                    http_send_text(client, 500, "health response too large");
                    close(client);
                    continue;
                }
                if (http_send_header(client, 200, "application/json", blen))
                    http_send(client, body, blen);
            } else {
                const char *msg = "not found";
                http_send_text(client, 404, msg);
            }
        } else if (strcmp(method, "POST") == 0 &&
                   (strcmp(path, "/chat/completions") == 0 || strcmp(path, "/v1/chat/completions") == 0)) {
            /* Find body after \r\n\r\n */
            char *body = strstr(req, "\r\n\r\n");
            if (!body) {
                http_send_text(client, 400, "bad request");
                close(client);
                continue;
            }
            body += 4;

            char user_msg[2048] = "";
            int user_msg_len = json_get_last_user_message(body, user_msg, sizeof(user_msg));

            if (user_msg_len < 0) {
                const char *err = "{\"error\":\"invalid user message\"}";
                if (http_send_header(client, 400, "application/json", (int)strlen(err)))
                    http_send(client, err, (int)strlen(err));
            } else if (user_msg[0] == '\0') {
                const char *err = "{\"error\":\"no user message\"}";
                if (http_send_header(client, 400, "application/json", (int)strlen(err)))
                    http_send(client, err, (int)strlen(err));
            } else {
                float temp = json_get_float(body, "temperature", 0.0f);
                float max_tok_f = json_get_float(body, "max_tokens", 256.0f);
                int max_tok = 256;
                if (max_tok_f >= 1.0f) max_tok = max_tok_f > 512.0f ? 512 : (int)max_tok_f;
                printf("[serve] inference: \"%.*s\" temp=%.2f max=%d\n",
                       (int)(strlen(user_msg) > 60 ? 60 : strlen(user_msg)), user_msg, temp, max_tok);
                if (http_send_header(client, 200, "text/event-stream", -1))
                    http_stream_inference(client, ps, user_msg, temp, max_tok);
            }
        } else {
            const char *msg = "method not allowed";
            http_send_text(client, 400, msg);
        }

        close(client);
    }
}

static int doe_parse_long_arg(const char *flag, const char *value, long *out) {
    if (!value || !out) {
        fprintf(stderr, "[doe] missing %s value\n", flag ? flag : "argument");
        return 0;
    }
    errno = 0;
    char *end = NULL;
    long v = strtol(value, &end, 10);
    while (end && (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r')) end++;
    if (end == value || errno || !end || *end != '\0') {
        fprintf(stderr, "[doe] invalid %s value: %s\n", flag ? flag : "argument", value);
        return 0;
    }
    *out = v;
    return 1;
}

static int doe_parse_float_arg(const char *flag, const char *value, float *out) {
    if (!value || !out) {
        fprintf(stderr, "[doe] missing %s value\n", flag ? flag : "argument");
        return 0;
    }
    errno = 0;
    char *end = NULL;
    float v = strtof(value, &end);
    while (end && (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r')) end++;
    if (end == value || errno || !end || *end != '\0' || !isfinite(v)) {
        fprintf(stderr, "[doe] invalid %s value: %s\n", flag ? flag : "argument", value);
        return 0;
    }
    *out = v;
    return 1;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MAIN — the field manifests.
 * ═══════════════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    printf("\n  doe.c — Democracy of Experts\n");
    printf("  θ = ε + γ + αδ — the parliament awakens.\n\n");

    char gguf_path[256] = "";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i+1 < argc) snprintf(gguf_path, 256, "%s", argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i+1 < argc) {
            long v; if (!doe_parse_long_arg("--threads", argv[++i], &v)) return 1;
            if (v < 1) v = 1;
            if (v > 32) v = 32;
            g_n_threads = (int)v;
        }
        else if (strcmp(argv[i], "--rep-penalty") == 0 && i+1 < argc) {
            float v; if (!doe_parse_float_arg("--rep-penalty", argv[++i], &v)) return 1;
            g_rep_penalty = v;
            if (g_rep_penalty <= 0.0f) g_rep_penalty = 1.0f;
        }
        else if (strcmp(argv[i], "--field-gain") == 0 && i+1 < argc) {
            float v; if (!doe_parse_float_arg("--field-gain", argv[++i], &v)) return 1;
            g_field_gain = v;
            if (g_field_gain < 0.0f) g_field_gain = 0.0f;
        }
        else if (strcmp(argv[i], "--train") == 0 && i+1 < argc) {
            long v; if (!doe_parse_long_arg("--train", argv[++i], &v)) return 1;
            g_train = v ? 1 : 0;
        }
        else if (strcmp(argv[i], "--max-new") == 0 && i+1 < argc) {
            long v; if (!doe_parse_long_arg("--max-new", argv[++i], &v)) return 1;
            if (v < 1) v = 1;
            if (v > 512) v = 512;
            g_gen_max_new = (int)v;
        }
        else if (strcmp(argv[i], "--top-k") == 0 && i+1 < argc) {
            long v; if (!doe_parse_long_arg("--top-k", argv[++i], &v)) return 1;
            if (v < 0) v = 0;
            if (v > INT_MAX) v = INT_MAX;
            g_gen_top_k = (int)v;
        }
        else if (strcmp(argv[i], "--temp") == 0 && i+1 < argc) {
            float v; if (!doe_parse_float_arg("--temp", argv[++i], &v)) return 1;
            g_gen_temp_override = v;
            if (g_gen_temp_override < 0.0f) g_gen_temp_override = -1.0f;
        }
        else if (strcmp(argv[i], "--once") == 0) { g_once = 1; }
        else if (strcmp(argv[i], "--rope-norm") == 0) { g_rope_norm = 1; }
        else if (strcmp(argv[i], "--no-load-spore") == 0) { g_load_spore = 0; }
        else if (strcmp(argv[i], "--no-save-spore") == 0) { g_save_spore = 0; }
        else if (strcmp(argv[i], "--prophecy") == 0 && i+1 < argc) { /* will be set after field_init */ }
        else if (strcmp(argv[i], "--destiny") == 0 && i+1 < argc) { /* will be set after field_init */ }
        else if (strcmp(argv[i], "--serve") == 0 && i+1 < argc) {
            long v; if (!doe_parse_long_arg("--serve", argv[++i], &v)) return 1;
            if (v < 1 || v > 65535) {
                fprintf(stderr, "[doe] invalid --serve value: %ld (expected 1..65535)\n", v);
                return 1;
            }
            g_serve_port = (int)v;
        }
        else if (strcmp(argv[i], "--serve-public") == 0) { g_serve_public = 1; }
        else if (strcmp(argv[i], "--image") == 0 && i+1 < argc) snprintf(g_image_path, sizeof g_image_path, "%s", argv[++i]);
        else if (strcmp(argv[i], "--mmproj") == 0 && i+1 < argc) snprintf(g_mmproj_path, sizeof g_mmproj_path, "%s", argv[++i]);
        else if (strcmp(argv[i], "--img-embeds-bin") == 0 && i+1 < argc) snprintf(g_img_embeds_bin, sizeof g_img_embeds_bin, "%s", argv[++i]);
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("doe.c — DOE: inference architecture over any GGUF\n\n");
            printf("  --model PATH    GGUF to index (or auto-detect)\n");
            printf("  --serve PORT    start HTTP server for UI (doe_ui.html, doe.html, yent.html, worldmodel.html)\n");
            printf("  --threads N     CPU threads for matvec (default: all cores)\n");
            printf("  --prophecy N    prediction horizon (default: 7)\n");
            printf("  --destiny F     destiny bias strength (default: 0.35)\n");
            printf("  --lora-alpha F  LoRA injection strength (default: 0.1)\n");
            printf("  --rep-penalty F repetition penalty over GENERATED tokens (default: 1.0 = off; 1.05 polishes demo tail)\n");
            printf("  --field-gain F  scale Dario field overlay on logits (default: 1.0; 0 = bare host)\n");
            printf("  --train N       notorch online-learning during inference (default: 0 = off; 1 = on)\n\n");
            printf("  --max-new N     max generated tokens in chat mode (default: 200)\n");
            printf("  --top-k N       chat sampler top-k (default: 40; 1 with --temp 0 matches greedy probes)\n");
            printf("  --temp F        override chat sampler temperature (default: field temperature; 0 = greedy)\n");
            printf("  --once          exit after one generated answer (for artifact-isolation probes)\n\n");
            printf("  --rope-norm     probe llama.cpp NORM RoPE (consecutive head pairs)\n");
            printf("  --no-load-spore skip mycelium restore for isolated probes\n");
            printf("  --no-save-spore skip mycelium persist on exit for isolated probes\n\n");
            printf("  BLAS: cc doe.c -O3 -lm -lpthread -DUSE_BLAS -DACCELERATE -framework Accelerate -o doe\n");
            printf("  GPU:  cc doe.c -O3 -lm -lpthread -DUSE_CUBLAS -lcublas -lcudart -o doe\n");
            return 0;
        }
    }

    /* ── Thread count for matvec ── */
    if (g_n_threads == 0) g_n_threads = (int)sysconf(_SC_NPROCESSORS_ONLN); /* D-L1: respect --threads if the user set it */
    if (g_n_threads < 1) g_n_threads = 1;
    if (g_n_threads > 32) g_n_threads = 32;

    /* ── Field awakens ── */
    field_init();

    /* Parse field overrides */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--prophecy") == 0 && i+1 < argc) {
            long v; if (!doe_parse_long_arg("--prophecy", argv[++i], &v)) return 1;
            if (v < INT_MIN || v > INT_MAX) {
                fprintf(stderr, "[doe] invalid --prophecy value: %ld (outside int range)\n", v);
                return 1;
            }
            F.prophecy = (int)v;
        }
        else if (strcmp(argv[i], "--destiny") == 0 && i+1 < argc) {
            float v; if (!doe_parse_float_arg("--destiny", argv[++i], &v)) return 1;
            F.destiny = v;
        }
        else if (strcmp(argv[i], "--lora-rank") == 0 && i+1 < argc) { i++; /* D-L1: accepted for compat; rank comes from the GGUF, not a flag */ }
        else if (strcmp(argv[i], "--lora-alpha") == 0 && i+1 < argc) {
            float v; if (!doe_parse_float_arg("--lora-alpha", argv[++i], &v)) return 1;
            F.lora_alpha = v;
        }
    }

    /* ── Environment scan ── */
    Environment env;
    env_scan(&env, __FILE__);

    /* ── PHASE 1: Search for DOE identity + gamma FIRST ── */
    char identity_path[256] = "";
    char gamma_path[256] = "";
    int weightless = 1;
    {
        static const char *wdirs[] = { "weights/", "doe_w/", "./", "../weights/", NULL };
        struct stat st;
        /* Search for doe_identity*.gguf (any variant: _micro, _mini, _q8, etc.) */
        for (int d = 0; wdirs[d] && identity_path[0] == '\0'; d++) {
            DIR *dir = opendir(wdirs[d]);
            if (!dir) continue;
            struct dirent *ent;
            int64_t best_size = 0;
            while ((ent = readdir(dir)) != NULL) {
                if (strncmp(ent->d_name, "doe_identity", 12) != 0) continue;
                int nlen = (int)strlen(ent->d_name);
                if (nlen < 5 || strcmp(ent->d_name + nlen - 5, ".gguf") != 0) continue;
                char tmp[256];
                snprintf(tmp, 256, "%s%s", wdirs[d], ent->d_name);
                if (stat(tmp, &st) == 0 && st.st_size > best_size) {
                    snprintf(identity_path, 256, "%s", tmp);
                    best_size = st.st_size;
                }
            }
            closedir(dir);
            if (identity_path[0] != '\0') {
                stat(identity_path, &st);
                printf("[identity] found: %s (%.1fMB)\n", identity_path, (float)st.st_size/(1024*1024));
                weightless = 0;
            }
        }
        /* Search for doe_gamma*.bin or doe_gamma*.npz */
        for (int d = 0; wdirs[d] && gamma_path[0] == '\0'; d++) {
            DIR *dir = opendir(wdirs[d]);
            if (!dir) continue;
            struct dirent *ent;
            while ((ent = readdir(dir)) != NULL) {
                if (strncmp(ent->d_name, "doe_gamma", 9) == 0 ||
                    strncmp(ent->d_name, "gamma_", 6) == 0) {
                    char tmp[256];
                    snprintf(tmp, 256, "%s%s", wdirs[d], ent->d_name);
                    if (stat(tmp, &st) == 0 && st.st_size > 0) {
                        snprintf(gamma_path, 256, "%s", tmp);
                        printf("[gamma] found: %s (%.1fMB)\n", tmp, (float)st.st_size/(1024*1024));
                        break;
                    }
                }
            }
            closedir(dir);
        }
        if (weightless)
            printf("[identity] no doe_identity.gguf — weightless mode.\n");
        if (gamma_path[0] == '\0')
            printf("[gamma] no doe_gamma.bin — parliament self-organizes.\n");
    }

    /* ── PHASE 2: Find host GGUF (external knowledge substrate) ── */
    if (gguf_path[0] == '\0') {
        if (identity_path[0] != '\0') {
            snprintf(gguf_path, 256, "%s", identity_path);
            printf("[host] using identity as host model.\n");
        } else {
            /* Also check all discovered GGUFs for doe.identity metadata */
            int identity_idx = -1, external_idx = -1;
            for (int i = 0; i < env.n_ggufs; i++) {
                if (strstr(env.ggufs[i].path, "mycelium/")) continue;
                if (strstr(env.ggufs[i].path, "doe_gamma")) continue;
                /* Quick sniff for doe.identity key in this GGUF */
                if (strstr(env.ggufs[i].path, "doe_identity")) {
                    identity_idx = i; continue;
                }
                if (external_idx < 0) external_idx = i;
            }
            /* Identity GGUF by name takes priority */
            if (identity_idx >= 0) {
                snprintf(gguf_path, 256, "%s", env.ggufs[identity_idx].path);
                printf("[host] found identity GGUF: %s\n", gguf_path);
                weightless = 0;
            } else if (external_idx >= 0) {
                snprintf(gguf_path, 256, "%s", env.ggufs[external_idx].path);
                printf("[host] indexing external: %s (%.1fMB)\n", gguf_path, (float)env.ggufs[external_idx].file_size/(1024*1024));
            } else {
                fprintf(stderr, "[error] no GGUF found. use --model PATH or place a .gguf nearby.\n");
                return 1;
            }
        }
    }

    /* ── Index GGUF ── */
    GGUFIndex idx;
    if (!index_load(&idx, gguf_path)) {
        fprintf(stderr, "[error] failed to index %s\n", gguf_path);
        return 1;
    }
    idx.weightless = weightless;

    /* If GGUF has doe.identity metadata — it's ours regardless of filename */
    if (idx.identity_tag[0] != '\0') {
        idx.weightless = 0;
        printf("[identity] verified via metadata: \"%s\"\n", idx.identity_tag);
    }

    /* ── Load gamma if found ── */
    if (gamma_path[0] != '\0') {
        errno = 0;
        FILE *gf = fopen(gamma_path, "rb");
        if (gf) {
            errno = 0;
            if (fseek(gf, 0, SEEK_END) != 0) {
                int seek_errno = errno;
                printf("[gamma] cannot seek %s%s%s\n", gamma_path,
                       seek_errno ? ": " : "", seek_errno ? strerror(seek_errno) : "");
            } else {
                errno = 0;
                long gsz = ftell(gf);
                int ftell_errno = errno;
                if (gsz < 0) {
                    printf("[gamma] cannot size %s%s%s\n", gamma_path,
                           ftell_errno ? ": " : "", ftell_errno ? strerror(ftell_errno) : "");
                } else if (gsz == 0 || gsz > INT_MAX) {
                    printf("[gamma] invalid gamma size for %s: %ld bytes\n", gamma_path, gsz);
                } else {
                    errno = 0;
                    if (fseek(gf, 0, SEEK_SET) != 0) {
                        int rewind_errno = errno;
                        printf("[gamma] cannot rewind %s%s%s\n", gamma_path,
                               rewind_errno ? ": " : "", rewind_errno ? strerror(rewind_errno) : "");
                    } else {
                        errno = 0;
                        idx.gamma_data = malloc((size_t)gsz);
                        if (!idx.gamma_data) {
                            int malloc_errno = errno;
                            printf("[gamma] allocation failed for %s (%ld bytes)%s%s\n", gamma_path, gsz,
                                   malloc_errno ? ": " : "", malloc_errno ? strerror(malloc_errno) : "");
                        } else {
                            size_t want = (size_t)gsz;
                            errno = 0;
                            size_t got = fread(idx.gamma_data, 1, want, gf);
                            int read_errno = errno;
                            if (got == want) {
                                idx.gamma_size = (int)gsz;
                                printf("[gamma] loaded %ld bytes — personality active.\n", gsz);
                            } else {
                                if (ferror(gf)) {
                                    printf("[gamma] read error from %s after %zu/%zu bytes%s%s\n",
                                           gamma_path, got, want,
                                           read_errno ? ": " : "", read_errno ? strerror(read_errno) : "");
                                } else {
                                    printf("[gamma] short read from %s (%zu/%zu bytes)\n",
                                           gamma_path, got, want);
                                }
                                free(idx.gamma_data);
                                idx.gamma_data = NULL;
                                idx.gamma_size = 0;
                            }
                        }
                    }
                }
            }
            fclose(gf);
        } else {
            int open_errno = errno;
            printf("[gamma] cannot open %s%s%s\n", gamma_path,
                   open_errno ? ": " : "", open_errno ? strerror(open_errno) : "");
        }
    }

    /* ── Mycelium — check for existing LoRA spores ── */
    MyceliumState mycelium;
    mycelium_init(&mycelium);
    if (g_load_spore && mycelium_load(&idx, idx.profile.fingerprint))
        printf("[mycelium] resumed adaptation for this index.\n");

#ifdef USE_METAL
    /* Parliament-on-GPU: pack the (now final, post-spore) frozen experts into a
     * registered arena so the election/inject kernels run inside the resident
     * command buffer at alpha!=0 without dropping x off the GPU. Metal-live +
     * parliament-active only; experts are static at --train 0. */
    if (nt_metal_available() && idx.lora_alpha != 0.0f) field_repack_arena(&idx);
#endif

    /* ── Chat or Serve ── */
    if (g_serve_port > 0) {
        /* Resolve directory of the executable for HTML files */
        char exe_dir[512] = "./";
        {
            /* Try to find doe_ui.html relative to argv[0] */
            char *slash = strrchr(argv[0], '/');
            if (slash) { int dlen = (int)(slash - argv[0]) + 1; if (dlen < 500) { memcpy(exe_dir, argv[0], dlen); exe_dir[dlen] = '\0'; } }
        }
        serve_loop(&idx, exe_dir);
    } else {
        chat(&idx);
    }

    /* ── Save spore on exit ── */
    if (g_save_spore) mycelium_save(&idx, F.step, F.field_health);

    /* ── Cleanup ── */
    index_free(&idx);
    printf("[doe] the parliament adjourns. θ persists.\n");
    return 0;
}
