import json
import re
import torch

DATASET_FILE = "dataset.txt"
VOCAB_FILE = "vocab.json"
OUTPUT_FILE = "train_data.pt"

# -----------------------------
# cargar vocabulario
# -----------------------------

with open(
    VOCAB_FILE,
    "r",
    encoding="utf-8"
) as f:

    vocab = json.load(f)

UNK = vocab["<unk>"]

# -----------------------------
# leer dataset
# -----------------------------

with open(
    DATASET_FILE,
    "r",
    encoding="utf-8"
) as f:

    text = f.read().lower()

# preservar tokens especiales

text = text.replace("<usuario>", " <usuario> ")
text = text.replace("<asistente>", " <asistente> ")
text = text.replace("<fin>", " <fin> ")

# exactamente el mismo patrón que tokenizer.py

tokens = re.findall(
    r"<usuario>|<asistente>|<fin>|[a-z0-9\\.\\-]+",
    text
)

print("tokens encontrados:", len(tokens))

# -----------------------------
# convertir a ids
# -----------------------------

token_ids = []

for token in tokens:

    if token in vocab:
        token_ids.append(vocab[token])
    else:
        token_ids.append(UNK)

print("token ids:", len(token_ids))

# -----------------------------
# guardar tensor
# -----------------------------

tensor = torch.tensor(
    token_ids,
    dtype=torch.long
)

torch.save(
    tensor,
    OUTPUT_FILE
)

print()
print("archivo generado:", OUTPUT_FILE)
print("shape:", tensor.shape)

print()
print("primeros 50 token ids:")
print(token_ids[:50])
