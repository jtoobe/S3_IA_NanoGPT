/*
 * NanoGPT en ESP32-S3 - inferencia INT8
 * Compatible con entrenar_modelo5.py (dataset.json)
 *
 * Formato de secuencia que el entrenador construye:
 *   <usuario> pregunta <asistente> respuesta <fin>
 *   índices:  0=<pad> 1=<unk> 2=<usuario> 3=<asistente> 4=<fin>
 * Programacion asistida con Qwen3.7-Plus
 */

#include "model.h"
#include "vocab.h"
#include <math.h>
#include <string.h>

// ===== Debe coincidir con entrenar_modelo5.py =====
#define EMB_SIZE   48
#define BLOCK_SIZE 24
#define NUM_HEADS  4
#define HEAD_SIZE  (EMB_SIZE / NUM_HEADS)   // 12
#define NUM_BLOCKS 2
#define FF_HIDDEN  96
#define MAX_NEW_TOKENS 15

static_assert(NUM_BLOCKS == 2, "Este sketch soporta 2 bloques");
static_assert(NUM_HEADS  == 4, "Este sketch soporta 4 cabezas");
static_assert(EMB_SIZE % NUM_HEADS == 0, "EMB_SIZE debe divisible por NUM_HEADS");

#define TOK_PAD 0
#define TOK_UNK 1
#define TOK_USR 2
#define TOK_ASI 3
#define TOK_FIN 4

// ===== Pesos por bloque (cada cabeza tiene su propio array y escala) =====
struct BlockWeights {
  const int8_t *ln1w, *ln1b, *ln2w, *ln2b;
  float ln1ws, ln1bs, ln2ws, ln2bs;
  const int8_t *qw[NUM_HEADS], *kw[NUM_HEADS], *vw[NUM_HEADS];
  float qws[NUM_HEADS], kws[NUM_HEADS], vws[NUM_HEADS];
  const int8_t *pw, *pb, *f0w, *f0b, *f2w, *f2b;
  float pws, pbs, f0ws, f0bs, f2ws, f2bs;
};
static BlockWeights BW[NUM_BLOCKS];

#define ASSIGN_HEAD(B, H)                                                        \
  BW[B].qw[H]  = blocks_##B##_sa_heads_##H##_query_weight;                        \
  BW[B].qws[H] = blocks_##B##_sa_heads_##H##_query_weight_scale;                  \
  BW[B].kw[H]  = blocks_##B##_sa_heads_##H##_key_weight;                          \
  BW[B].kws[H] = blocks_##B##_sa_heads_##H##_key_weight_scale;                    \
  BW[B].vw[H]  = blocks_##B##_sa_heads_##H##_value_weight;                        \
  BW[B].vws[H] = blocks_##B##_sa_heads_##H##_value_weight_scale;

#define ASSIGN_BLOCK(B)                                                          \
  BW[B].ln1w  = blocks_##B##_ln1_weight;  BW[B].ln1ws = blocks_##B##_ln1_weight_scale; \
  BW[B].ln1b  = blocks_##B##_ln1_bias;    BW[B].ln1bs = blocks_##B##_ln1_bias_scale;   \
  BW[B].ln2w  = blocks_##B##_ln2_weight;  BW[B].ln2ws = blocks_##B##_ln2_weight_scale; \
  BW[B].ln2b  = blocks_##B##_ln2_bias;    BW[B].ln2bs = blocks_##B##_ln2_bias_scale;   \
  BW[B].pw    = blocks_##B##_sa_proj_weight; BW[B].pws = blocks_##B##_sa_proj_weight_scale; \
  BW[B].pb    = blocks_##B##_sa_proj_bias;   BW[B].pbs = blocks_##B##_sa_proj_bias_scale;   \
  BW[B].f0w   = blocks_##B##_ff_net_0_weight; BW[B].f0ws = blocks_##B##_ff_net_0_weight_scale; \
  BW[B].f0b   = blocks_##B##_ff_net_0_bias;   BW[B].f0bs = blocks_##B##_ff_net_0_bias_scale;   \
  BW[B].f2w   = blocks_##B##_ff_net_2_weight; BW[B].f2ws = blocks_##B##_ff_net_2_weight_scale; \
  BW[B].f2b   = blocks_##B##_ff_net_2_bias;   BW[B].f2bs = blocks_##B##_ff_net_2_bias_scale;   \
  ASSIGN_HEAD(B, 0) ASSIGN_HEAD(B, 1) ASSIGN_HEAD(B, 2) ASSIGN_HEAD(B, 3)

// ===== Buffers globales (~23 KB, evita stack overflow) =====
static int   tokens[BLOCK_SIZE];
static int   ntok = 0;
static float X [BLOCK_SIZE][EMB_SIZE];   // estado residual
static float LN[BLOCK_SIZE][EMB_SIZE];   // ln1(x) de TODAS las posiciones
static float Qb[BLOCK_SIZE][EMB_SIZE];
static float Kb[BLOCK_SIZE][EMB_SIZE];
static float Vb[BLOCK_SIZE][EMB_SIZE];
static float scores[BLOCK_SIZE];
static float ctx[EMB_SIZE];
static float proj[EMB_SIZE];
static float tmp[EMB_SIZE];
static float ffh[FF_HIDDEN];
static float ffo[EMB_SIZE];

String inputText;

// ===== Normalización idéntica a limpiar_texto() de Python =====
// Python: lower -> conservar solo alfanumérico/espacio -> quitar tildes
String normalizeWord(String w) {
  w.toLowerCase();
  w.replace("á", "a"); w.replace("é", "e"); w.replace("í", "i");
  w.replace("ó", "o"); w.replace("ú", "u"); w.replace("ñ", "n");
  String out;
  out.reserve(w.length());
  for (size_t i = 0; i < w.length(); i++) {
    char c = w[i];
    if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == ' ') out += c;
  }
  out.trim();
  return out;
}

int findToken(const String& word) {
  for (int i = 5; i < VOCAB_SIZE; i++)          // salta tokens especiales
    if (word.equalsIgnoreCase(VOCAB[i])) return i;
  return TOK_UNK;
}

// ===== Operaciones =====
static void layerNorm(float* v, const int8_t* w, float ws,
                      const int8_t* b, float bs, int n) {
  float mean = 0.f, var = 0.f;
  for (int i = 0; i < n; i++) mean += v[i];
  mean /= n;
  for (int i = 0; i < n; i++) { float d = v[i] - mean; var += d * d; }
  var /= n;
  float inv = 1.0f / sqrtf(var + 1e-5f);
  for (int i = 0; i < n; i++)
    v[i] = ((v[i] - mean) * inv) * (w[i] * ws) + (b[i] * bs);
}

static void matmul(const float* in, const int8_t* W, float ws,
                   float* out, int in_n, int out_n) {
  for (int o = 0; o < out_n; o++) {
    const int8_t* row = W + (size_t)o * in_n;
    float s = 0.f;
    for (int i = 0; i < in_n; i++) s += in[i] * (float)row[i];
    out[o] = s * ws;
  }
}

// GELU exacta (igual que F.gelu de PyTorch, no la aproximación tanh)
static inline float gelu(float z) {
  return 0.5f * z * (1.0f + erff(z * 0.7071067811865476f));
}

// ===== Forward completo: réplica de NanoGPT.forward() =====
static void forward(int n) {
  // 1) embeddings de token + posición, siempre desde los IDs
  for (int t = 0; t < n; t++) {
    const int8_t* te = token_embedding_weight + (size_t)tokens[t] * EMB_SIZE;
    const int8_t* pe = position_embedding_weight + (size_t)t * EMB_SIZE;
    for (int i = 0; i < EMB_SIZE; i++)
      X[t][i] = te[i] * token_embedding_weight_scale
              + pe[i] * position_embedding_weight_scale;
  }

  const float inv_sqrt_hs = 1.0f / sqrtf((float)HEAD_SIZE);

  for (int b = 0; b < NUM_BLOCKS; b++) {
    const BlockWeights& w = BW[b];

    // 2) ln1 de TODAS las posiciones (antes de tocar X)
    for (int t = 0; t < n; t++) {
      memcpy(LN[t], X[t], sizeof(LN[t]));
      layerNorm(LN[t], w.ln1w, w.ln1ws, w.ln1b, w.ln1bs, EMB_SIZE);
    }

    // 3) Q, K, V de TODAS las posiciones desde ln1
    for (int t = 0; t < n; t++) {
      for (int h = 0; h < NUM_HEADS; h++) {
        int off = h * HEAD_SIZE;
        matmul(LN[t], w.qw[h], w.qws[h], &Qb[t][off], EMB_SIZE, HEAD_SIZE);
        matmul(LN[t], w.kw[h], w.kws[h], &Kb[t][off], EMB_SIZE, HEAD_SIZE);
        matmul(LN[t], w.vw[h], w.vws[h], &Vb[t][off], EMB_SIZE, HEAD_SIZE);
      }
      yield();
    }

    // 4) atención causal + residual
    for (int t = 0; t < n; t++) {
      memset(ctx, 0, sizeof(ctx));
      for (int h = 0; h < NUM_HEADS; h++) {
        int off = h * HEAD_SIZE;
        float maxs = -1e30f;
        for (int i = 0; i <= t; i++) {
          float s = 0.f;
          for (int j = 0; j < HEAD_SIZE; j++)
            s += Qb[t][off + j] * Kb[i][off + j];
          s *= inv_sqrt_hs;
          scores[i] = s;
          if (s > maxs) maxs = s;
        }
        float sum = 0.f;
        for (int i = 0; i <= t; i++) { scores[i] = expf(scores[i] - maxs); sum += scores[i]; }
        float invsum = 1.0f / sum;
        for (int i = 0; i <= t; i++) {
          float a = scores[i] * invsum;
          for (int j = 0; j < HEAD_SIZE; j++) ctx[off + j] += a * Vb[i][off + j];
        }
      }
      matmul(ctx, w.pw, w.pws, proj, EMB_SIZE, EMB_SIZE);
      for (int i = 0; i < EMB_SIZE; i++) X[t][i] += proj[i] + w.pb[i] * w.pbs;
      yield();
    }

    // 5) feed-forward + residual
    for (int t = 0; t < n; t++) {
      memcpy(tmp, X[t], sizeof(tmp));
      layerNorm(tmp, w.ln2w, w.ln2ws, w.ln2b, w.ln2bs, EMB_SIZE);
      matmul(tmp, w.f0w, w.f0ws, ffh, EMB_SIZE, FF_HIDDEN);
      for (int i = 0; i < FF_HIDDEN; i++) ffh[i] = gelu(ffh[i] + w.f0b[i] * w.f0bs);
      matmul(ffh, w.f2w, w.f2ws, ffo, FF_HIDDEN, EMB_SIZE);
      for (int i = 0; i < EMB_SIZE; i++) X[t][i] += ffo[i] + w.f2b[i] * w.f2bs;
      yield();
    }
  }
}

// ===== argmax sobre TODO el vocabulario (igual que Python) =====
static int argmaxNext(int n) {
  memcpy(tmp, X[n - 1], sizeof(tmp));
  layerNorm(tmp, ln_f_weight, ln_f_weight_scale, ln_f_bias, ln_f_bias_scale, EMB_SIZE);

  float best = -1e30f;
  int bestTok = TOK_UNK;
  for (int v = 0; v < VOCAB_SIZE; v++) {
    const int8_t* row = lm_head_weight + (size_t)v * EMB_SIZE;
    float s = 0.f;
    for (int i = 0; i < EMB_SIZE; i++) s += tmp[i] * (float)row[i];
    s = s * lm_head_weight_scale + lm_head_bias[v] * lm_head_bias_scale;
    if (s > best) { best = s; bestTok = v; }
  }
  return bestTok;
}

// ===== Generación =====
void generar(String txt) {
  ntok = 0;
  tokens[ntok++] = TOK_USR;                 // <usuario> lo agrega el entrenador

  int start = 0;
  while (ntok < BLOCK_SIZE - 2) {
    int pos = txt.indexOf(' ', start);
    String raw = (pos == -1) ? txt.substring(start) : txt.substring(start, pos);
    String word = normalizeWord(raw);
    if (word.length() > 0) {
      // una "palabra" puede quedar vacía tras limpiar (ej. "-")
      int sp = word.indexOf(' ');
      if (sp == -1) tokens[ntok++] = findToken(word);
      else {
        tokens[ntok++] = findToken(word.substring(0, sp));
        if (ntok < BLOCK_SIZE - 2) tokens[ntok++] = findToken(word.substring(sp + 1));
      }
    }
    if (pos == -1) break;
    start = pos + 1;
  }
  tokens[ntok++] = TOK_ASI;                 // <asistente>

  Serial.print("\n>> "); Serial.println(txt);
  Serial.print("   tokens:");
  for (int t = 0; t < ntok; t++) { Serial.print(' '); Serial.print(tokens[t]); }
  Serial.println();
  Serial.print(">> ");

  unsigned long t0 = millis();
  int limite = min(BLOCK_SIZE, ntok + MAX_NEW_TOKENS);

  while (ntok < limite) {
    forward(ntok);                        // SIEMPRE desde los IDs
    int next = argmaxNext(ntok);
    if (next <= TOK_FIN) break;           // <pad>/<unk>/<usuario>/<asistente>/<fin>
    Serial.print(VOCAB[next]); Serial.print(' ');
    tokens[ntok++] = next;
  }

  Serial.print("  ["); Serial.print(millis() - t0); Serial.println(" ms]");
}

// ===== Verificación de coherencia modelo/sketch =====
bool checkModel() {
  bool ok = true;
  if (position_embedding_weight_size != BLOCK_SIZE * EMB_SIZE) {
    Serial.printf("ERROR: position_embedding=%d, esperado %d (BLOCK_SIZE*EMB_SIZE)\n",
                  (int)position_embedding_weight_size, BLOCK_SIZE * EMB_SIZE);
    ok = false;
  }
  if (token_embedding_weight_size != VOCAB_SIZE * EMB_SIZE) {
    Serial.printf("ERROR: token_embedding=%d, esperado %d (VOCAB_SIZE*EMB_SIZE)\n",
                  (int)token_embedding_weight_size, VOCAB_SIZE * EMB_SIZE);
    ok = false;
  }
  if (blocks_0_ff_net_0_weight_size != FF_HIDDEN * EMB_SIZE) {
    Serial.printf("ERROR: ff_net_0=%d, esperado %d (FF_HIDDEN*EMB_SIZE)\n",
                  (int)blocks_0_ff_net_0_weight_size, FF_HIDDEN * EMB_SIZE);
    ok = false;
  }
  if (blocks_0_sa_heads_0_query_weight_size != HEAD_SIZE * EMB_SIZE) {
    Serial.printf("ERROR: head query=%d, esperado %d (HEAD_SIZE*EMB_SIZE)\n",
                  (int)blocks_0_sa_heads_0_query_weight_size, HEAD_SIZE * EMB_SIZE);
    ok = false;
  }
  return ok;
}

void setup() {
  Serial.begin(115200);
  delay(2000);

  ASSIGN_BLOCK(0)
  ASSIGN_BLOCK(1)

  Serial.println("\n===========================================");
  Serial.println("  NanoGPT ESP32-S3 (dataset.json)");
  Serial.printf("  EMB=%d HEADS=%d BLOCKS=%d FF=%d BLOCK_SIZE=%d\n",
                EMB_SIZE, NUM_HEADS, NUM_BLOCKS, FF_HIDDEN, BLOCK_SIZE);
  Serial.printf("  VOCAB_SIZE=%d\n", VOCAB_SIZE);
  Serial.println("===========================================");

  if (!checkModel()) {
    Serial.println("Las dimensiones del .ino NO coinciden con model.h");
    Serial.println("Ajusta los #define y recompila.");
    while (true) delay(1000);
  }
  Serial.println("Modelo verificado. Escribi una frase:");
}

void loop() {
  if (Serial.available()) {
    inputText = Serial.readStringUntil('\n');
    inputText.trim();
    if (inputText.length() > 0) generar(inputText);
  }
}
