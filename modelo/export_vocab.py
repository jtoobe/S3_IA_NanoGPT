import json

with open(
    "vocab.json",
    "r",
    encoding="utf-8"
) as f:
    vocab = json.load(f)

reverse = sorted(
    vocab.items(),
    key=lambda x: x[1]
)

with open(
    "vocab.h",
    "w",
    encoding="utf-8"
) as f:

    f.write("#pragma once\n\n")

    f.write(
        f"const int VOCAB_SIZE = {len(reverse)};\n\n"
    )

    f.write(
        "const char* VOCAB[] = {\n"
    )

    for word, idx in reverse:

        word = word.replace('\"','\\\"')

        f.write(
            f'    "{word}",\n'
        )

    f.write("};\n")

print("vocab.h generado")
