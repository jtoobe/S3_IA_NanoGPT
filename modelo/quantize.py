import torch

INPUT_MODEL = "nanogpt_words.pth"
OUTPUT_MODEL = "nanogpt_words_int8.pth"

print("Cargando modelo...")

weights = torch.load(
    INPUT_MODEL,
    map_location="cpu"
)

quantized = {}

for name, tensor in weights.items():

    max_abs = tensor.abs().max()

    if max_abs == 0:
        scale = 1.0
    else:
        scale = max_abs / 127.0

    q = torch.round(
        tensor / scale
    ).to(torch.int8)

    quantized[name] = {
        "scale": float(scale),
        "weight": q
    }

    print(
        f"{name:40s}",
        tensor.shape
    )

torch.save(
    quantized,
    OUTPUT_MODEL
)

print()
print("Guardado:", OUTPUT_MODEL)
