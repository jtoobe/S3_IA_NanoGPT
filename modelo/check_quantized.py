import torch

model = torch.load(
    "nanogpt_words_int8.pth",
    map_location="cpu"
)

for name,data in model.items():

    print(
        name,
        data["weight"].dtype,
        data["weight"].shape,
        data["scale"]
    )

    break
