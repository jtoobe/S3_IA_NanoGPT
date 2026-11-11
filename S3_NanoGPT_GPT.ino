/*
  Chat local ESP32-S3 - NanoGPT INT8
  Modelo: model.h | Vocabulario: vocab.h
  Consola: Monitor Serie a 115200 baudios, fin de linea "Nueva linea".

  IMPORTANTE:
  - El modelo fue entrenado con secuencias de 16 tokens.
  - La tokenizacion original no fue entregada. Este sketch usa tokens separados
    por espacios, en minusculas y sin acentos. Es una aproximacion.
  - Los pesos INT8 se convierten a float como int8 * escala del tensor.
  - Generacion greedy (elige siempre el token de mayor logit).
*/

#include <Arduino.h>
#include <math.h>
#include <string.h>
#include "model.h"
#include "vocab.h"

static const int N_EMBD = 32;
static const int N_HEAD = 2;
static const int HEAD_SIZE = 16;
static const int N_LAYER = 2;
static const int BLOCK_SIZE = 16;
static const int FF_SIZE = 128;
static const int MAX_INPUT_CHARS = 240;
static const int MAX_NEW_TOKENS = 15;

int tokenIds[BLOCK_SIZE];
int tokenCount = 0;

float xbuf[BLOCK_SIZE][N_EMBD];
float norm1[BLOCK_SIZE][N_EMBD];
float norm2[BLOCK_SIZE][N_EMBD];
float qbuf[BLOCK_SIZE][HEAD_SIZE];
float kbuf[BLOCK_SIZE][HEAD_SIZE];
float vbuf[BLOCK_SIZE][HEAD_SIZE];
float attnConcat[BLOCK_SIZE][N_EMBD];
float attnOut[BLOCK_SIZE][N_EMBD];
float ffHidden[FF_SIZE];
float ffOut[N_EMBD];
float finalNorm[N_EMBD];
float logits[VOCAB_SIZE];
float scores[BLOCK_SIZE];

int findToken(const char *word) {
  for (int i = 0; i < VOCAB_SIZE; ++i) {
    if (strcmp(word, VOCAB[i]) == 0) return i;
  }
  return 1; // <unk>
}

void normalizeText(char *s) {
  // Normalizacion sencilla para coincidir mejor con el vocabulario ASCII.
  for (char *p = s; *p; ++p) {
    if (*p >= 'A' && *p <= 'Z') *p = *p - 'A' + 'a';
  }
}

int tokenize(const char *text, int *dst, int maxTokens) {
  char work[MAX_INPUT_CHARS + 1];
  strncpy(work, text, MAX_INPUT_CHARS);
  work[MAX_INPUT_CHARS] = '\0';
  normalizeText(work);

  int n = 0;
  char *saveptr = nullptr;
  char *word = strtok_r(work, " \t\r\n", &saveptr);
  while (word && n < maxTokens) {
    dst[n++] = findToken(word);
    word = strtok_r(nullptr, " \t\r\n", &saveptr);
  }
  return n;
}

inline float deq(const int8_t *w, float scale, int i) {
  return ((float)w[i]) * scale;
}

void layerNorm(const float *in, float *out,
               const int8_t *weight, float weightScale,
               const int8_t *bias, float biasScale) {
  float mean = 0.0f;
  for (int i = 0; i < N_EMBD; ++i) mean += in[i];
  mean /= (float)N_EMBD;

  float var = 0.0f;
  for (int i = 0; i < N_EMBD; ++i) {
    float d = in[i] - mean;
    var += d * d;
  }
  var /= (float)N_EMBD;

  float invStd = 1.0f / sqrtf(var + 1.0e-5f);
  for (int i = 0; i < N_EMBD; ++i) {
    float gamma = deq(weight, weightScale, i);
    float beta = deq(bias, biasScale, i);
    out[i] = (in[i] - mean) * invStd * gamma + beta;
  }
}

void linear32(const float *in, float *out,
              const int8_t *weight, float weightScale,
              const int8_t *bias, float biasScale,
              int outDim, int inDim) {
  // PyTorch nn.Linear: matriz [out_features, in_features], row-major.
  for (int o = 0; o < outDim; ++o) {
    float sum = bias ? deq(bias, biasScale, o) : 0.0f;
    const int base = o * inDim;
    for (int i = 0; i < inDim; ++i) {
      sum += in[i] * deq(weight, weightScale, base + i);
    }
    out[o] = sum;
  }
}

void runAttention(int layer) {
  // Normalizacion de entrada para todas las posiciones del contexto.
  for (int t = 0; t < tokenCount; ++t) {
    if (layer == 0) {
      layerNorm(xbuf[t], norm1[t],
        blocks_0_ln1_weight, blocks_0_ln1_weight_scale,
        blocks_0_ln1_bias, blocks_0_ln1_bias_scale);
    } else {
      layerNorm(xbuf[t], norm1[t],
        blocks_1_ln1_weight, blocks_1_ln1_weight_scale,
        blocks_1_ln1_bias, blocks_1_ln1_bias_scale);
    }
  }

  // Multi-head self-attention causal: cada posicion solo mira el pasado.
  for (int h = 0; h < N_HEAD; ++h) {
    const int8_t *kw, *qw, *vw;
    float ks, qs, vs;
    if (layer == 0 && h == 0) {
      kw=blocks_0_sa_heads_0_key_weight; ks=blocks_0_sa_heads_0_key_weight_scale;
      qw=blocks_0_sa_heads_0_query_weight; qs=blocks_0_sa_heads_0_query_weight_scale;
      vw=blocks_0_sa_heads_0_value_weight; vs=blocks_0_sa_heads_0_value_weight_scale;
    } else if (layer == 0 && h == 1) {
      kw=blocks_0_sa_heads_1_key_weight; ks=blocks_0_sa_heads_1_key_weight_scale;
      qw=blocks_0_sa_heads_1_query_weight; qs=blocks_0_sa_heads_1_query_weight_scale;
      vw=blocks_0_sa_heads_1_value_weight; vs=blocks_0_sa_heads_1_value_weight_scale;
    } else if (layer == 1 && h == 0) {
      kw=blocks_1_sa_heads_0_key_weight; ks=blocks_1_sa_heads_0_key_weight_scale;
      qw=blocks_1_sa_heads_0_query_weight; qs=blocks_1_sa_heads_0_query_weight_scale;
      vw=blocks_1_sa_heads_0_value_weight; vs=blocks_1_sa_heads_0_value_weight_scale;
    } else {
      kw=blocks_1_sa_heads_1_key_weight; ks=blocks_1_sa_heads_1_key_weight_scale;
      qw=blocks_1_sa_heads_1_query_weight; qs=blocks_1_sa_heads_1_query_weight_scale;
      vw=blocks_1_sa_heads_1_value_weight; vs=blocks_1_sa_heads_1_value_weight_scale;
    }

    // Proyecciones Q, K y V para cada token.
    for (int t = 0; t < tokenCount; ++t) {
      linear32(norm1[t], qbuf[t], qw, qs, nullptr, 0, HEAD_SIZE, N_EMBD);
      linear32(norm1[t], kbuf[t], kw, ks, nullptr, 0, HEAD_SIZE, N_EMBD);
      linear32(norm1[t], vbuf[t], vw, vs, nullptr, 0, HEAD_SIZE, N_EMBD);
    }

    for (int t = 0; t < tokenCount; ++t) {
      float maxScore = -1.0e30f;
      for (int j = 0; j <= t; ++j) {
        float dot = 0.0f;
        for (int d = 0; d < HEAD_SIZE; ++d) dot += qbuf[t][d] * kbuf[j][d];
        scores[j] = dot * (1.0f / sqrtf((float)HEAD_SIZE));
        if (scores[j] > maxScore) maxScore = scores[j];
      }
      float denom = 0.0f;
      for (int j = 0; j <= t; ++j) {
        scores[j] = expf(scores[j] - maxScore);
        denom += scores[j];
      }
      for (int d = 0; d < HEAD_SIZE; ++d) {
        float value = 0.0f;
        for (int j = 0; j <= t; ++j) value += (scores[j] / denom) * vbuf[j][d];
        attnConcat[t][h * HEAD_SIZE + d] = value;
      }
    }
  }

  // Proyeccion de salida de la atencion.
  const int8_t *projW, *projB;
  float projWS, projBS;
  if (layer == 0) {
    projW=blocks_0_sa_proj_weight; projWS=blocks_0_sa_proj_weight_scale;
    projB=blocks_0_sa_proj_bias; projBS=blocks_0_sa_proj_bias_scale;
  } else {
    projW=blocks_1_sa_proj_weight; projWS=blocks_1_sa_proj_weight_scale;
    projB=blocks_1_sa_proj_bias; projBS=blocks_1_sa_proj_bias_scale;
  }
  for (int t = 0; t < tokenCount; ++t) {
    linear32(attnConcat[t], attnOut[t], projW, projWS, projB, projBS, N_EMBD, N_EMBD);
    for (int d = 0; d < N_EMBD; ++d) xbuf[t][d] += attnOut[t][d];
  }
}

void runFeedForward(int layer) {
  const int8_t *lnW, *lnB, *ff0W, *ff0B, *ff2W, *ff2B;
  float lnWS, lnBS, ff0WS, ff0BS, ff2WS, ff2BS;

  if (layer == 0) {
    lnW=blocks_0_ln2_weight; lnWS=blocks_0_ln2_weight_scale;
    lnB=blocks_0_ln2_bias; lnBS=blocks_0_ln2_bias_scale;
    ff0W=blocks_0_ff_net_0_weight; ff0WS=blocks_0_ff_net_0_weight_scale;
    ff0B=blocks_0_ff_net_0_bias; ff0BS=blocks_0_ff_net_0_bias_scale;
    ff2W=blocks_0_ff_net_2_weight; ff2WS=blocks_0_ff_net_2_weight_scale;
    ff2B=blocks_0_ff_net_2_bias; ff2BS=blocks_0_ff_net_2_bias_scale;
  } else {
    lnW=blocks_1_ln2_weight; lnWS=blocks_1_ln2_weight_scale;
    lnB=blocks_1_ln2_bias; lnBS=blocks_1_ln2_bias_scale;
    ff0W=blocks_1_ff_net_0_weight; ff0WS=blocks_1_ff_net_0_weight_scale;
    ff0B=blocks_1_ff_net_0_bias; ff0BS=blocks_1_ff_net_0_bias_scale;
    ff2W=blocks_1_ff_net_2_weight; ff2WS=blocks_1_ff_net_2_weight_scale;
    ff2B=blocks_1_ff_net_2_bias; ff2BS=blocks_1_ff_net_2_bias_scale;
  }

  for (int t = 0; t < tokenCount; ++t) {
    layerNorm(xbuf[t], norm2[t], lnW, lnWS, lnB, lnBS);
    linear32(norm2[t], ffHidden, ff0W, ff0WS, ff0B, ff0BS, FF_SIZE, N_EMBD);
    for (int i = 0; i < FF_SIZE; ++i) if (ffHidden[i] < 0.0f) ffHidden[i] = 0.0f; // ReLU
    linear32(ffHidden, ffOut, ff2W, ff2WS, ff2B, ff2BS, N_EMBD, FF_SIZE);
    for (int d = 0; d < N_EMBD; ++d) xbuf[t][d] += ffOut[d];
  }
}

int predictNext() {
  if (tokenCount <= 0) return 1;

  // Embedding de tokens + embedding posicional.
  for (int t = 0; t < tokenCount; ++t) {
    int id = tokenIds[t];
    for (int d = 0; d < N_EMBD; ++d) {
      xbuf[t][d] =
        deq(token_embedding_weight, token_embedding_weight_scale, id * N_EMBD + d) +
        deq(position_embedding_weight, position_embedding_weight_scale, t * N_EMBD + d);
    }
  }

  runAttention(0);
  runFeedForward(0);
  runAttention(1);
  runFeedForward(1);

  layerNorm(xbuf[tokenCount - 1], finalNorm,
    ln_f_weight, ln_f_weight_scale, ln_f_bias, ln_f_bias_scale);

  linear32(finalNorm, logits,
    lm_head_weight, lm_head_weight_scale,
    lm_head_bias, lm_head_bias_scale,
    VOCAB_SIZE, N_EMBD);

  int best = 0;
  for (int i = 1; i < VOCAB_SIZE; ++i) if (logits[i] > logits[best]) best = i;
  return best;
}

void appendToken(int id) {
  if (tokenCount >= BLOCK_SIZE) {
    // Ventana deslizante: conserva los ultimos 15 tokens.
    for (int i = 1; i < BLOCK_SIZE; ++i) tokenIds[i - 1] = tokenIds[i];
    tokenCount = BLOCK_SIZE - 1;
  }
  tokenIds[tokenCount++] = id;
}

void chat(const char *question) {
  tokenCount = 0;
  appendToken(2); // <usuario>

  int userTokens[BLOCK_SIZE];
  int n = tokenize(question, userTokens, BLOCK_SIZE - 2);
  for (int i = 0; i < n; ++i) appendToken(userTokens[i]);

  appendToken(3); // <asistente>

  Serial.print("\nAsistente: ");
  int generated = 0;
  while (generated < MAX_NEW_TOKENS) {
    int next = predictNext();
    if (next == 4 || next == 0) break; // <fin> o <pad>
    appendToken(next);
    if (strcmp(VOCAB[next], "<unk>") != 0) {
      Serial.print(VOCAB[next]);
      Serial.print(' ');
    } else {
      Serial.print("[?] ");
    }
    ++generated;
  }
  if (generated == 0) Serial.print("(sin respuesta; prueba otra pregunta)");
  Serial.println();
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  delay(1200);
  Serial.println();
  Serial.println("==================================");
  Serial.println(" Chat ESP32-S3 - NanoGPT INT8");
  Serial.println(" Escribe una pregunta y presiona ENTER");
  Serial.println(" Ejemplo: que es esp32");
  Serial.println("==================================");
}

void loop() {
  static char input[MAX_INPUT_CHARS + 1];
  static size_t pos = 0;

  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      input[pos] = '\0';
      if (pos > 0) chat(input);
      pos = 0;
      Serial.print("> ");
    } else if (pos < MAX_INPUT_CHARS) {
      input[pos++] = c;
    }
  }
}
