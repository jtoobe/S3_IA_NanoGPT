import torch

INPUT_MODEL = "nanogpt_words_int8.pth"
OUTPUT_HEADER = "model.h"

print("Cargando modelo INT8...")

model = torch.load(
    INPUT_MODEL,
    map_location="cpu"
)

with open(
    OUTPUT_HEADER,
    "w",
    encoding="utf-8"
) as f:

    f.write("#pragma once\n\n")
    f.write("// Modelo NanoGPT INT8 exportado\n\n")

    for name, data in model.items():

        scale = data["scale"]
        weight = data["weight"]

        array_name = (
            name
            .replace(".", "_")
            .replace(":", "_")
        )

        flat = weight.flatten().tolist()

        f.write(
            f"const float {array_name}_scale = {scale}f;\n\n"
        )

        f.write(
            f"const int32_t {array_name}_size = {len(flat)};\n\n"
        )

        f.write(
            f"const int8_t {array_name}[] = {{\n"
        )

        for i, value in enumerate(flat):

            f.write(str(value))

            if i != len(flat) - 1:
                f.write(",")

            if (i + 1) % 32 == 0:
                f.write("\n")

        f.write("\n};\n\n")

print()
print("Archivo generado:")
print(OUTPUT_HEADER)
