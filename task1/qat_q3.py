import sys
import json
import math
import os
import numpy as np
import torch

sys.path.insert(0, "nanoGPT")
from model import GPT, GPTConfig
from qat_layers import QATLinear
from quant_core import quantize_weight_per_channel

checkpoint_path = "task1/results/q3_qat_checkpoint.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
model = GPT(GPTConfig(**checkpoint["model_args"]))
model.load_state_dict(checkpoint["model"])

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

with open(checkpoint["activation_ranges"], "r", encoding="utf-8") as file:
    all_params = json.load(file)

converted_names = []

for name, layer in list(model.named_modules()):
    if not isinstance(layer, torch.nn.Linear):
        continue
    params = all_params[name]
    parent_path, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    setattr(parent, child_name, QATLinear(layer, params["scale"], params["zero_point"]))
    converted_names.append(name)

assert len(converted_names) == 17
model.eval()
torch.backends.cuda.matmul.allow_tf32 = False

quantized_weights = {}
with torch.no_grad():
    for name, layer in model.named_modules():  # 再次遍历模型，查找 QATLinear
        if isinstance(layer, QATLinear):  # 只导出已量化的 Linear 层
            weight_q, weight_scale = quantize_weight_per_channel(layer.weight)  # 将训练后的权重逐通道量化
            quantized_weights[name] = {
                "weight": weight_q.cpu(),
                "scale": weight_scale.cpu(),
                "zero_point": 0,
            }

weights_path = "task1/results/q3_weights.pt"
torch.save({"weight_scheme": "symmetric_int8_per_channel", "layers": quantized_weights}, weights_path)
weights_file_bytes = os.path.getsize(weights_path)

val_data = np.memmap("nanoGPT/data/shakespeare_char/val.bin", dtype=np.uint16, mode="r")
batch_size = 32  # 与前三组评估一样，每批取 32 段
block_size = 128  # 与前三组评估一样，每段取 128 个字符编号
eval_batches = 100  # 与前三组评估一样，共评估 100 批
generator = torch.Generator().manual_seed(1337)
starts_all = torch.randint(len(val_data) - block_size, (eval_batches, batch_size), generator=generator)
losses = []

with torch.no_grad():  # 验证时不更新权重，也不需要记录梯度
    for batch_starts in starts_all:
        starts = batch_starts.tolist()
        inputs = np.stack([val_data[i:i + block_size] for i in starts]).astype(np.int64)
        targets = np.stack([val_data[i + 1:i + block_size + 1] for i in starts]).astype(np.int64)
        x = torch.tensor(inputs, dtype=torch.long, device=device)
        y = torch.tensor(targets, dtype=torch.long, device=device)
        _, loss = model(x, y)
        losses.append(loss.item())

mean_loss = sum(losses) / len(losses)  # 计算 100 批验证 loss 的平均值
ppl = math.exp(mean_loss)  # 根据平均 loss 计算困惑度 PPL

metrics = {
    "run_tag": "Q3",
    "checkpoint": checkpoint_path,
    "source_checkpoint": checkpoint["source_checkpoint"],
    "calibration": checkpoint["activation_ranges"],
    "qat_steps": checkpoint["train_steps"],
    "weight_quantization": "symmetric_int8_per_channel",
    "activation_quantization": "asymmetric_int8_per_tensor_static",
    "eval_seed": 1337,
    "eval_batches": eval_batches,
    "batch_size": batch_size,
    "block_size": block_size,
    "validation_tokens": eval_batches * batch_size * block_size,
    "quantized_linear_layers": len(converted_names),
    "validation_loss": mean_loss,
    "validation_ppl": ppl,
    "weight_artifact": weights_path,
    "weight_file_bytes": weights_file_bytes,
}

result_path = "task1/results/q3_metrics.json"
with open(result_path, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

print("QAT steps:", checkpoint["train_steps"])
print("validation tokens:", metrics["validation_tokens"])
print("Q3 validation loss:", mean_loss)
print("Q3 validation PPL:", ppl)
print("Q3 weights saved to:", weights_path, "bytes:", weights_file_bytes)
print("Q3 metrics saved to:", result_path)