import sys
import math
import numpy as np
import torch
import json

sys.path.insert(0, "nanoGPT")
from model import GPT, GPTConfig

from quant_core import quantize_symmetric, dequantize_symmetric,fake_quantize_with_params

run_tag = "Q1"
weight_scheme = "symmetric_int8_per_tensor"

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

model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())  # 单独复制输出层权重，避免同时改动与它共享权重的 embedding

layer_names = {}  # 保存“Linear 对象 → 层名称”的对应关系
handles = []


def fake_quant_input(module, inputs):
    name = layer_names[module]  # 找到当前 Linear 的名称
    params = all_params[name]  # 读取该层固定的校准参数
    x_hat = fake_quantize_with_params(inputs[0], params["scale"], params["zero_point"])
    return (x_hat,)


for name, layer in model.named_modules():  # 遍历模型的所有模块
    if isinstance(layer, torch.nn.Linear):  # 只处理 Linear
        layer_names[layer] = name  # 记录这个模块的名称

        with torch.no_grad():  # 仅修改推理时内存中的权重
            weight_q, weight_scale = quantize_symmetric(layer.weight)  # 这一层权重共用一个 scale
            weight_hat = dequantize_symmetric(weight_q, weight_scale)  # 得到带 INT8 误差的浮点权重
            layer.weight.copy_(weight_hat)  # 放回这一层

        handles.append(layer.register_forward_pre_hook(fake_quant_input))  # 登记该层的输入量化 hook

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

for handle in handles:  # 逐个处理已登记的 hook
    handle.remove()  # 取消 hook

mean_loss = sum(losses) / len(losses)
ppl = math.exp(mean_loss)

print("device:", device)
print("validation tokens:", eval_batches * batch_size * block_size)
print("validation loss:", mean_loss)
print("validation PPL:", ppl)
print("mode: Q1 per-tensor W8A8 fake quant")  # 标记本次实验
print("quantized Linear layers:", len(handles))

metrics = {
    "run_tag": run_tag,
    "checkpoint": "nanoGPT/out/task0-bf16/best_ckpt.pt",
    "calibration": "task1/results/activation_ranges.json",
    "weight_quantization": weight_scheme,
    "activation_quantization": "asymmetric_int8_per_tensor_static",
    "eval_seed": 1337,
    "eval_batches": eval_batches,
    "batch_size": batch_size,
    "block_size": block_size,
    "validation_tokens": eval_batches * batch_size * block_size,
    "quantized_linear_layers": len(handles),
    "validation_loss": mean_loss,
    "validation_ppl": ppl,
}

result_path = f"task1/results/{run_tag.lower()}_metrics.json"
with open(result_path, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2, ensure_ascii=False)
print("metrics saved to:", result_path)