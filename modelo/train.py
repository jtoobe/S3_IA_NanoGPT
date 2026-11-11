import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------
# Configuracion
# ---------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16
BLOCK_SIZE = 16

MAX_ITERS = 10000

LEARNING_RATE = 1e-3

N_EMBD = 32
N_HEAD = 2
N_LAYER = 2

#VOCAB_SIZE = 375

import json

with open(
    "vocab.json",
    "r",
    encoding="utf-8"
) as f:
    vocab = json.load(f)

VOCAB_SIZE = len(vocab)

print("VOCAB_SIZE =", VOCAB_SIZE)

# ---------------------------------
# Dataset
# ---------------------------------

data = torch.load("train_data.pt")

print("dataset:", data.shape)

# 90% entrenamiento
split = int(len(data) * 0.9)

train_data = data[:split]
val_data = data[split:]

# ---------------------------------
# Batch
# ---------------------------------

def get_batch(split_name):

    source = train_data if split_name == "train" else val_data

    ix = torch.randint(
        len(source) - BLOCK_SIZE - 1,
        (BATCH_SIZE,)
    )

    x = torch.stack([
        source[i:i + BLOCK_SIZE]
        for i in ix
    ])

    y = torch.stack([
        source[i + 1:i + BLOCK_SIZE + 1]
        for i in ix
    ])

    return (
        x.to(DEVICE),
        y.to(DEVICE)
    )

# ---------------------------------
# Head
# ---------------------------------

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

        wei = q @ k.transpose(-2, -1)

        wei = wei * (
            k.shape[-1] ** -0.5
        )

        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        wei = F.softmax(
            wei,
            dim=-1
        )

        v = self.value(x)

        return wei @ v

# ---------------------------------
# Multi Head
# ---------------------------------

class MultiHead(nn.Module):

    def __init__(self):

        super().__init__()

        head_size = N_EMBD // N_HEAD

        self.heads = nn.ModuleList([
            Head(head_size)
            for _ in range(N_HEAD)
        ])

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

# ---------------------------------
# Feed Forward
# ---------------------------------

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

# ---------------------------------
# Transformer Block
# ---------------------------------

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

# ---------------------------------
# NanoGPT
# ---------------------------------

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

        self.blocks = nn.Sequential(*[
            Block()
            for _ in range(N_LAYER)
        ])

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

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits = logits.view(
                B * T,
                C
            )

            targets = targets.view(
                B * T
            )

            loss = F.cross_entropy(
                logits,
                targets
            )

        return logits, loss

# ---------------------------------
# Entrenamiento
# ---------------------------------

model = NanoGPT().to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

print(model)

for step in range(MAX_ITERS):

    xb, yb = get_batch("train")

    logits, loss = model(
        xb,
        yb
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if step % 100 == 0:

        print(
            step,
            loss.item()
        )

torch.save(
    model.state_dict(),
    "nanogpt_words.pth"
)

print()
print("modelo guardado")
print("nanogpt_words.pth")
