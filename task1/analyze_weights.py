import sys
import json
import torch

sys.path.insert(0, "nanoGPT")
from model import GPT, GPTConfig
from quant_core import quantize_symmetric, dequantize_symmetric
from quant_core import quantize_weight_per_channel, dequantize_weight_per_channel

checkpoint = torch.load("nanoGPT/out/task0-bf16/best_ckpt.pt", map_location="cpu", weights_only=False)
model = GPT(GPTConfig(**checkpoint["model_args"]))
model.load_state_dict(checkpoint["model"])
model.eval()

weight_stats = {}  # 按层名保存权重统计与量化误差

with torch.no_grad(): 
    for name, layer in model.named_modules():
        if not isinstance(layer, torch.nn.Linear):
            continue
        w = layer.weight.detach().to(torch.float32)  # 取出该层的原始权重，并确保按 FP32 统计
        q1, q1_scale = quantize_symmetric(w)  # 用整个矩阵共用的 scale 量化权重
        q1_hat = dequantize_symmetric(q1, q1_scale)  # 将 Q1 整数权重还原成近似浮点值
        q2, q2_scale = quantize_weight_per_channel(w)  # 用每个输出行自己的 scale 量化权重
        q2_hat = dequantize_weight_per_channel(q2, q2_scale)  # 将 Q2 整数权重还原成近似浮点值
        weight_stats[name] = {  # 开始记录当前层的数据
            "shape": list(w.shape),
            "min": w.min().item(),  # 记录最小权重
            "max": w.max().item(),  # 记录最大权重
            "max_abs": w.abs().max().item(),  # 记录绝对值最大的权重
            "mean": w.mean().item(),  # 记录所有权重的平均值
            "std": w.std(unbiased=False).item(),  # 记录总体标准差，表示数值的离散程度
            "rms": w.square().mean().sqrt().item(),  # 先平方、求平均、开方，得到 RMS
            "channel_max_abs": w.abs().amax(dim=1).tolist(),  # 记录每个输出行的最大绝对值
            "q1_mean_abs_error": (w - q1_hat).abs().mean().item(),  # 记录 Q1 的平均绝对量化误差
            "q2_mean_abs_error": (w - q2_hat).abs().mean().item(),  # 记录 Q2 的平均绝对量化误差
            "q1_max_abs_error": (w - q1_hat).abs().max().item(),  # 记录 Q1 的最大绝对量化误差
            "q2_max_abs_error": (w - q2_hat).abs().max().item(),  # 记录 Q2 的最大绝对量化误差
            "fp32_weight_bytes": w.numel() * w.element_size(),  # 计算该层原始 FP32 权重的数据字节数
            "int8_weight_bytes": q1.numel() * q1.element_size(),  # 计算该层 INT8 权重的数据字节数
            "q1_scale_bytes": q1_scale.numel() * q1_scale.element_size(),  # 计算 Q1 的 scale 数据字节数
            "q2_scale_bytes": q2_scale.numel() * q2_scale.element_size(),  # 计算 Q2 的 scale 数据字节数
        }

result_path = "task1/results/weight_stats.json"
with open(result_path, "w", encoding="utf-8") as file:
    json.dump(weight_stats, file, indent=2)

example = weight_stats["transformer.h.0.mlp.c_fc"]
print("Linear layers:", len(weight_stats))
print("example Q1 mean error:", example["q1_mean_abs_error"])
print("example Q2 mean error:", example["q2_mean_abs_error"])
print("saved to:", result_path)