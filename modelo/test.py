import json
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================
# Configuracion
# =====================================

DEVICE = "cpu"

BLOCK_SIZE = 16

N_EMBD = 32
N_HEAD = 2
N_LAYER = 2

# =====================================
# Cargar vocabulario
# =====================================

with open(
    "vocab.json",
    "r",
    encoding="utf-8"
) as f:

    vocab = json.load(f)

itos = {
    v: k
    for k, v in vocab.items()
}

VOCAB_SIZE = len(vocab)

# =====================================
# Head
# =====================================

class Head(nn.Module):

    def __init__(self, head_size):

        super().__init__()

        self.key = nn.Linear(
            N_EMBD,
            head_size,
            bias=False
        )

        self.query = nn.Linear(
            N_EMBD,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            N_EMBD,
            head_size,
            bias=False
        )

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(
                    BLOCK_SIZE,
                    BLOCK_SIZE
                )
            )
        )

    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2,-1)

        wei = wei * (
            k.shape[-1] ** -0.5
        )

        wei = wei.masked_fill(
            self.tril[:T,:T] == 0,
            float("-inf")
        )

        wei = F.softmax(
            wei,
            dim=-1
        )

        v = self.value(x)

        return wei @ v

# =====================================
# MultiHead
# =====================================

class MultiHead(nn.Module):

    def __init__(self):

        super().__init__()

        head_size = N_EMBD // N_HEAD

        self.heads = nn.ModuleList(
            [
                Head(head_size)
                for _ in range(N_HEAD)
            ]
        )

        self.proj = nn.Linear(
            N_EMBD,
            N_EMBD
        )

    def forward(self, x):

        out = torch.cat(
            [h(x) for h in self.heads],
            dim=-1
        )

        return self.proj(out)

# =====================================
# FeedForward
# =====================================

class FeedForward(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                N_EMBD,
                4 * N_EMBD
            ),
            nn.ReLU(),
            nn.Linear(
                4 * N_EMBD,
                N_EMBD
            )
        )

    def forward(self, x):

        return self.net(x)

# =====================================
# Block
# =====================================

class Block(nn.Module):

    def __init__(self):

        super().__init__()

        self.sa = MultiHead()

        self.ff = FeedForward()

        self.ln1 = nn.LayerNorm(
            N_EMBD
        )

        self.ln2 = nn.LayerNorm(
            N_EMBD
        )

    def forward(self, x):

        x = x + self.sa(
            self.ln1(x)
        )

        x = x + self.ff(
            self.ln2(x)
        )

        return x

# =====================================
# NanoGPT
# =====================================

class NanoGPT(nn.Module):

    def __init__(self):

        super().__init__()

        self.token_embedding = nn.Embedding(
            VOCAB_SIZE,
            N_EMBD
        )

        self.position_embedding = nn.Embedding(
            BLOCK_SIZE,
            N_EMBD
        )

        self.blocks = nn.Sequential(
            *[
                Block()
                for _ in range(N_LAYER)
            ]
        )

        self.ln_f = nn.LayerNorm(
            N_EMBD
        )

        self.lm_head = nn.Linear(
            N_EMBD,
            VOCAB_SIZE
        )

    def forward(
        self,
        idx,
        targets=None
    ):

        B, T = idx.shape

        tok = self.token_embedding(idx)

        pos = self.position_embedding(
            torch.arange(
                T,
                device=idx.device
            )
        )

        x = tok + pos

        x = self.blocks(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        return logits, None

# =====================================
# Cargar modelo
# =====================================

model = NanoGPT()

model.load_state_dict(
    torch.load(
        "nanogpt_words.pth",
        map_location="cpu"
    )
)

model.eval()

# =====================================
# Tokenizador
# =====================================

def tokenize(text):

    text = text.lower()

    tokens = re.findall(
        r"<usuario>|<asistente>|<fin>|[a-z0-9\\.\\-]+",
        text
    )

    result = []

    for token in tokens:

        if token in vocab:
            result.append(vocab[token])
        else:
            result.append(vocab["<unk>"])

    return result


# =====================================
# Chat
# =====================================

print()
print("NanoGPT ESP32 (palabras)")
print()

while True:

    pregunta = input("Pregunta: ").strip()

    if pregunta == "":
        break

    prompt = f"<usuario> {pregunta} <asistente>"

    ids = tokenize(prompt)

    idx = torch.tensor(
        [ids],
        dtype=torch.long
    )

    generated = ids.copy()

    for _ in range(50):

        idx_cond = idx[:, -BLOCK_SIZE:]

        with torch.no_grad():

            logits, _ = model(idx_cond)

        logits = logits[:, -1, :]

        # greedy decoding
        next_token = torch.argmax(
            logits,
            dim=-1
        ).item()

        generated.append(next_token)

        if next_token == vocab["<fin>"]:
            break

        idx = torch.tensor(
            [generated],
            dtype=torch.long
        )

    respuesta = []

    started = False

    for token_id in generated:

        word = itos[token_id]

        if word == "<asistente>":
            started = True
            continue

        if word == "<fin>":
            break

        if started:
            respuesta.append(word)

    print()
    print("Respuesta:")
    print(" ".join(respuesta))
    print()
