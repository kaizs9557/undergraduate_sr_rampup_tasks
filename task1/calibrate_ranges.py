import sys
import json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "nanoGPT")
from model import GPT, GPTConfig

checkpoint = torch.load(
    "nanoGPT/out/task0-bf16/best_ckpt.pt",
    map_location="cpu",
    weights_only=False,
)
model = GPT(GPTConfig(**checkpoint["model_args"]))
model.load_state_dict(checkpoint["model"])

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
torch.backends.cuda.matmul.allow_tf32 = False

linear_names = {}  # 用模块对象查找它的名称
ranges = {}        # 保存每层输入的最小值、最大值
handles = []       # 保存 hook 登记记录，结束时用于移除


def record_input(module, inputs):
    activation = inputs[0].detach()
    current_min = activation.min().item()
    current_max = activation.max().item()
    name = linear_names[module]

    if name not in ranges:
        ranges[name] = {"min": current_min, "max": current_max}
    else:
        ranges[name]["min"] = min(ranges[name]["min"], current_min)
        ranges[name]["max"] = max(ranges[name]["max"], current_max)


for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        linear_names[module] = name
        handle = module.register_forward_pre_hook(record_input)
        handles.append(handle)

train_data = np.memmap(
    "nanoGPT/data/shakespeare_char/train.bin",
    dtype=np.uint16,
    mode="r",
)

batch_size = 32
block_size = 128
calibration_batches = 128

generator = torch.Generator().manual_seed(1337)
starts_all = torch.randint(
    len(train_data) - block_size,
    (calibration_batches, batch_size),
    generator=generator,
)

with torch.no_grad():
    for batch_starts in starts_all:
        inputs = []
        targets = []

        for start in batch_starts.tolist():
            inputs.append(
                train_data[start:start + block_size].astype(np.int64)
            )
            targets.append(
                train_data[start + 1:start + block_size + 1].astype(np.int64)
            )

        x = torch.tensor(np.stack(inputs), dtype=torch.long, device=device)
        y = torch.tensor(np.stack(targets), dtype=torch.long, device=device)
        model(x, y)

for handle in handles:
    handle.remove()

for layer in ranges.values():  # 依次处理每个 Linear 层记录的 min/max
    range_min = min(layer["min"], 0.0)
    range_max = max(layer["max"], 0.0)

    if range_max == range_min:
        scale = 1.0
        zero_point = 0
    else:
        scale = (range_max - range_min) / 255
        zero_point = round(-128 - range_min / scale)
        zero_point = max(-128, min(127, zero_point))

    layer["scale"] = scale
    layer["zero_point"] = zero_point

output_dir = Path("task1/results")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "activation_ranges.json"

with output_path.open("w", encoding="utf-8") as file:
    json.dump(ranges, file, indent=2)

print("calibration batches:", calibration_batches)
print("recorded Linear layers:", len(ranges))
print("first MLP layer:", ranges["transformer.h.0.mlp.c_fc"])
print("saved to:", output_path)