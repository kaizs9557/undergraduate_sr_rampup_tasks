import sys
import math
import numpy as np
import torch
import json

sys.path.insert(0, "nanoGPT")
from model import GPT, GPTConfig

from quant_core import quantize_symmetric, dequantize_symmetric,fake_quantize_with_params

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

with open("task1/results/activation_ranges.json", "r", encoding="utf-8") as file:
    all_params = json.load(file)

layer_name = "transformer.h.0.mlp.c_fc"  # 本次只实验第一层 MLP 的第一个 Linear
params = all_params[layer_name]  # 取这层固定的 activation scale 和 zero-point
layer = model.transformer.h[0].mlp.c_fc  # 找到模型中对应的 Linear 对象

with torch.no_grad():
    weight_q, weight_scale = quantize_symmetric(layer.weight)
    weight_hat = dequantize_symmetric(weight_q, weight_scale)
    layer.weight.copy_(weight_hat)  # 把近似权重放进这一层


def fake_quant_input(module, inputs):
    x_hat = fake_quantize_with_params(
        inputs[0], params["scale"], params["zero_point"]
    )
    return (x_hat,)


handle = layer.register_forward_pre_hook(fake_quant_input)

val_data = np.memmap(
    "nanoGPT/data/shakespeare_char/val.bin",
    dtype=np.uint16,
    mode="r",
)

batch_size = 32
block_size = 128
eval_batches = 100

generator = torch.Generator().manual_seed(1337)
starts_all = torch.randint(
    len(val_data) - block_size,
    (eval_batches, batch_size),
    generator=generator,
)

losses = []

with torch.no_grad():
    for batch_starts in starts_all:
        inputs = []
        targets = []

        for start in batch_starts.tolist():
            inputs.append(
                val_data[start:start + block_size].astype(np.int64)
            )
            targets.append(
                val_data[start + 1:start + block_size + 1].astype(np.int64)
            )

        x = torch.tensor(np.stack(inputs), dtype=torch.long, device=device)
        y = torch.tensor(np.stack(targets), dtype=torch.long, device=device)

        _, loss = model(x, y)
        losses.append(loss.item())

handle.remove()  # 评估结束，取消这一层的输入替换
mean_loss = sum(losses) / len(losses)
ppl = math.exp(mean_loss)

print("device:", device)
print("validation tokens:", eval_batches * batch_size * block_size)
print("validation loss:", mean_loss)
print("validation PPL:", ppl)
print("quantized layer:", layer_name)  # 表明这是单层试验，不是完整 Q1