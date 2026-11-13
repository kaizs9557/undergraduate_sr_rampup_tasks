import sys
import json
import pickle
import numpy as np
import torch
sys.path.insert(0, "nanoGPT")
from model import GPT, GPTConfig
from quant_core import fake_quantize_with_params

checkpoint = torch.load("nanoGPT/out/task0-bf16/best_ckpt.pt", map_location="cpu", weights_only=False)
model = GPT(GPTConfig(**checkpoint["model_args"]))
model.load_state_dict(checkpoint["model"])
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
torch.backends.cuda.matmul.allow_tf32 = False

with open("task1/results/activation_ranges.json", "r", encoding="utf-8") as file:  # 打开原校准参数
    calibration = json.load(file)  # 读取每层原来的范围、scale 和零点
with open("task1/results/activation_stats.json", "r", encoding="utf-8") as file:  # 打开上一轮激活统计
    activation_stats = json.load(file)["layers"]  # 取得 17 层的统计字典
with open("nanoGPT/data/shakespeare_char/meta.pkl", "rb") as file:  # 打开字符编号对应表
    id_to_char = pickle.load(file)["itos"]  # 用编号查出对应的字符

vocab_size = model.config.vocab_size
layer_names = {}
records = {}
handles = []
context = {"tokens": None}

for name, layer in model.named_modules():
    if not isinstance(layer, torch.nn.Linear):
        continue
    layer_names[layer] = name
    threshold = activation_stats[name]["p99_9_abs_upper"]
    lower = max(min(calibration[name]["min"], 0.0), -threshold)  # 将原下界截到负阈值，并确保包含 0
    upper = min(max(calibration[name]["max"], 0.0), threshold)  # 将原上界截到正阈值，并确保包含 0
    if upper <= lower:  # 处理极端情况下全部数值相同的问题
        clipped_scale = 1.0
        clipped_zero_point = 0
    else:  # 正常情况下，根据截断后的范围重新计算量化参数
        clipped_scale = (upper - lower) / 255
        clipped_zero_point = round(-128 - lower / clipped_scale)
        clipped_zero_point = max(-128, min(127, clipped_zero_point))  # 把零点限制在 INT8 范围
    records[name] = {
        "threshold": threshold,  # 保存 P99.9 异常阈值
        "clipped_range": [lower, upper],  # 保存假设截断后的范围
        "clipped_scale": clipped_scale,  # 保存假设截断后的 scale
        "clipped_zero_point": clipped_zero_point,  # 保存假设截断后的零点
        "elements": 0,  # 记录激活数字总数
        "outlier_elements": 0,  # 记录超过阈值的激活数字数
        "positions": 0,  # 记录字符位置总数
        "outlier_positions": 0,  # 记录至少有一个异常通道的字符位置数
        "original_error_sum": 0.0,  # 累加原校准参数的绝对误差
        "clipped_error_sum": 0.0,  # 累加截断方案的绝对误差
        "token_positions": torch.zeros(vocab_size, dtype=torch.int64),  # 按字符编号统计出现位置数
        "token_outlier_positions": torch.zeros(vocab_size, dtype=torch.int64),  # 按字符编号统计异常位置数
        "channel_outlier_elements": torch.zeros(layer.in_features, dtype=torch.int64),  # 按通道统计异常数字数
    }

def record_input(layer, inputs):
    name = layer_names[layer]  # 找到当前层名
    record = records[name]  # 找到该层累计记录
    activation = inputs[0].detach().float()  # 取出输入激活，转成 FP32 用于统计
    mask = activation.abs() > record["threshold"]  # 标记绝对值超过 P99.9 的数字
    position_mask = mask.any(dim=-1).reshape(-1)  # 标记至少有一个异常通道的字符位置
    token_ids = context["tokens"].reshape(-1)  # 将当前批次字符编号排成一列
    record["elements"] += activation.numel()  # 累加激活数字总数
    record["outlier_elements"] += mask.sum().item()  # 累加异常激活数字数
    record["positions"] += token_ids.numel()  # 累加字符位置总数
    record["outlier_positions"] += position_mask.sum().item()  # 累加异常字符位置数
    record["token_positions"] += torch.bincount(token_ids, minlength=vocab_size).cpu()  # 按字符编号累计总位置数
    record["token_outlier_positions"] += torch.bincount(token_ids[position_mask], minlength=vocab_size).cpu()  # 按字符编号累计异常位置数
    record["channel_outlier_elements"] += mask.sum(dim=(0, 1)).to(torch.int64).cpu()  # 按输入通道累计异常数字数
    original = calibration[name]  # 取得这层原本使用的校准参数
    original_hat = fake_quantize_with_params(activation, original["scale"], original["zero_point"])  # 用原范围模拟量化
    clipped_hat = fake_quantize_with_params(activation, record["clipped_scale"], record["clipped_zero_point"])  # 用 P99.9 截断范围模拟量化
    record["original_error_sum"] += (activation - original_hat).abs().sum().item()  # 累加原方案误差
    record["clipped_error_sum"] += (activation - clipped_hat).abs().sum().item()  # 累加截断方案误差

for layer in layer_names:  # 遍历全部 17 个 Linear 层
    handles.append(layer.register_forward_pre_hook(record_input))  # 登记统计输入的 hook
assert len(handles) == 17

train_data = np.memmap("nanoGPT/data/shakespeare_char/train.bin", dtype=np.uint16, mode="r")
batch_size = 32
block_size = 128
calibration_batches = 128
generator = torch.Generator().manual_seed(1337)
starts_all = torch.randint(len(train_data) - block_size, (calibration_batches, batch_size), generator=generator)

with torch.no_grad():  # 只做统计，不训练模型
    for batch_starts in starts_all:  # 逐批处理固定的训练样本
        starts = batch_starts.tolist()
        inputs = np.stack([train_data[i:i + block_size] for i in starts]).astype(np.int64)
        targets = np.stack([train_data[i + 1:i + block_size + 1] for i in starts]).astype(np.int64)
        x = torch.tensor(inputs, dtype=torch.long, device=device)
        y = torch.tensor(targets, dtype=torch.long, device=device) 
        context["tokens"] = x
        model(x, y)

for handle in handles:
    handle.remove()

layer_results = {}  # 准备最终保存的逐层结果
for name, record in records.items():
    token_rows = []  # 保存该层每种字符的统计
    for token_id in range(vocab_size):
        total = record["token_positions"][token_id].item()  # 读取该字符出现的总位置数
        outliers = record["token_outlier_positions"][token_id].item()  # 读取该字符的异常位置数
        if total == 0:
            continue
        token_rows.append({"token_id": token_id, "character": id_to_char[token_id], "positions": total, "outlier_positions": outliers, "outlier_position_rate": outliers / total})  # 保存字符频率和异常率
    channel_rows = [{"channel": index, "outlier_elements": count} for index, count in enumerate(record["channel_outlier_elements"].tolist())]  # 逐通道整理异常数字数
    stable_tokens = [row for row in token_rows if row["positions"] >= 1000]  # 只用出现至少 1000 次的字符排名，避免极罕见字符误导
    layer_results[name] = {
        "p99_9_abs_threshold": record["threshold"],  # 保存异常判定阈值
        "original_scale": calibration[name]["scale"],  # 保存实际使用的 scale
        "clipped_range": record["clipped_range"],  # 保存假设截断范围
        "clipped_scale": record["clipped_scale"],  # 保存假设截断后的 scale
        "clipped_zero_point": record["clipped_zero_point"],  # 保存假设截断后的零点
        "scale_reduction_percent": 100 * (1 - record["clipped_scale"] / calibration[name]["scale"]),  # 计算 scale 缩小的百分比
        "outlier_element_fraction": record["outlier_elements"] / record["elements"],  # 计算异常数字占比
        "outlier_position_fraction": record["outlier_positions"] / record["positions"],  # 计算异常字符位置占比
        "original_mean_abs_error": record["original_error_sum"] / record["elements"],  # 计算原方案平均绝对误差
        "clipped_mean_abs_error": record["clipped_error_sum"] / record["elements"],  # 计算截断方案平均绝对误差
        "tokens": token_rows,  # 保存所有字符的明细，便于报告查阅
        "top_tokens_by_count": sorted(token_rows, key=lambda row: row["outlier_positions"], reverse=True)[:5],  # 找异常位置数最多的 5 个字符
        "top_tokens_by_rate": sorted(stable_tokens, key=lambda row: row["outlier_position_rate"], reverse=True)[:5],  # 找异常率最高的 5 个常见字符
        "top_channels": sorted(channel_rows, key=lambda row: row["outlier_elements"], reverse=True)[:5],  # 找异常数字最多的 5 个通道
    }

result = {"source_checkpoint": "nanoGPT/out/task0-bf16/best_ckpt.pt", "threshold": "absolute activation above histogram P99.9 upper edge", "calibration_batches": calibration_batches, "batch_size": batch_size, "block_size": block_size, "seed": 1337, "layers": layer_results}  # 保存实验方法及全部逐层结果
result_path = "task1/results/outlier_impact.json"
with open(result_path, "w", encoding="utf-8") as file:
    json.dump(result, file, indent=2, ensure_ascii=False)

example = layer_results["transformer.h.1.mlp.c_proj"]
print("analyzed Linear layers:", len(layer_results))
print("original scale:", example["original_scale"])  # 显示原 scale
print("clipped scale:", example["clipped_scale"])  # 显示截断后 scale
print("original mean error:", example["original_mean_abs_error"])  # 显示原平均绝对误差
print("clipped mean error:", example["clipped_mean_abs_error"])  # 显示截断后平均绝对误差
print("top tokens by rate:", example["top_tokens_by_rate"])  # 显示异常率最高的常见字符
print("saved to:", result_path)