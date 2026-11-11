import json
import re
from collections import Counter

DATASET_FILE = "dataset.txt"
VOCAB_FILE = "vocab.json"

print("Leyendo dataset...")

with open(
    DATASET_FILE,
    "r",
    encoding="utf-8"
) as f:

    text = f.read().lower()

# Mantener etiquetas especiales
text = text.replace("<usuario>", " <usuario> ")
text = text.replace("<asistente>", " <asistente> ")
text = text.replace("<fin>", " <fin> ")

# Extraer palabras
tokens = re.findall(
    r"<usuario>|<asistente>|<fin>|[a-z0-9\\.\\-]+",
    text
)

print(f"Tokens totales: {len(tokens)}")

# Contar frecuencia
counter = Counter(tokens)

# Tokens especiales
vocab = {
    "<pad>": 0,
    "<unk>": 1,
    "<usuario>": 2,
    "<asistente>": 3,
    "<fin>": 4
}

next_id = len(vocab)

for word, freq in counter.most_common():

    if word in vocab:
        continue

    vocab[word] = next_id
    next_id += 1

# Guardar vocabulario
with open(
    VOCAB_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        vocab,
        f,
        ensure_ascii=False,
        indent=2
    )

print()
print("Vocabulario generado")
print(f"Palabras únicas: {len(vocab)}")

print("\nPrimeros 30 tokens:\n")

for word, idx in list(vocab.items())[:30]:
    print(f"{idx:4d} -> {word}")
