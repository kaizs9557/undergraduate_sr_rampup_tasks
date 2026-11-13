import sys
import json
import math
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

with open("task1/results/activation_ranges.json", "r", encoding="utf-8") as file:
    calibration = json.load(file)

hist_bins = 4096  # 将绝对值范围分成 4096 个小区间
linear_names = {}  # 保存“层对象 → 层名”的对应关系
accumulators = {}  # 保存每层的数量、总和、平方和及误差总和
histograms = {}  # 保存每层的绝对值直方图
hist_limits = {}  # 保存每层直方图覆盖的绝对值上限
channel_maxima = {}  # 保存每个输入通道出现过的最大绝对值
top_events = {}  # 保存每批中最大的一个激活及其位置
handles = []  # 保存 hook 登记记录，结束时用于移除
context = {"tokens": None, "starts": None}  # 让 hook 知道当前批次的 token 和片段起点

for name, layer in model.named_modules():
    if isinstance(layer, torch.nn.Linear):
        linear_names[layer] = name
        accumulators[name] = {"count": 0, "sum": 0.0, "sum_sq": 0.0, "min": float("inf"), "max": float("-inf"), "error_sum": 0.0, "error_max": 0.0}
        histograms[name] = torch.zeros(hist_bins, dtype=torch.int64)  # 在 CPU 上创建 4096 个计数格
        bound = max(abs(calibration[name]["min"]), abs(calibration[name]["max"]), 1e-6)
        hist_limits[name] = bound * 1.0001  # 略微放宽上限，确保边界值也被计入
        channel_maxima[name] = torch.zeros(layer.in_features)  # 为每个输入通道创建最大值记录
        top_events[name] = []  # 创建该层异常位置记录列表

def record_input(layer, inputs):  # 在每个 Linear 计算之前调用，观察它收到的输入
    name = linear_names[layer]
    activation = inputs[0].detach().float()
    absolute = activation.abs()
    record = accumulators[name]
    record["count"] += activation.numel()
    record["sum"] += activation.sum().item()
    record["sum_sq"] += activation.square().sum().item()
    record["min"] = min(record["min"], activation.min().item())
    record["max"] = max(record["max"], activation.max().item())
    counts = torch.histc(absolute, bins=hist_bins, min=0, max=hist_limits[name])
    histograms[name] += counts.to(torch.int64).cpu()  # 把这批计数加到 CPU 上的总直方图
    batch_channel_max = absolute.amax(dim=(0, 1)).cpu()
    channel_maxima[name] = torch.maximum(channel_maxima[name], batch_channel_max)
    params = calibration[name]
    restored = fake_quantize_with_params(activation, params["scale"], params["zero_point"])
    error = (activation - restored).abs()
    record["error_sum"] += error.sum().item()
    record["error_max"] = max(record["error_max"], error.max().item())
    flat_index = absolute.reshape(-1).argmax().item()
    time_length = activation.shape[1]
    channel_count = activation.shape[2]
    batch_index = flat_index // (time_length * channel_count)
    position = (flat_index // channel_count) % time_length
    channel = flat_index % channel_count
    top_events[name].append({
        "abs_value": absolute[batch_index, position, channel].item(),
        "batch_start": context["starts"][batch_index].item(),
        "position": position,
        "channel": channel,
        "input_token_id": context["tokens"][batch_index, position].item(),
    })

for layer in linear_names:
    handles.append(layer.register_forward_pre_hook(record_input))

train_data = np.memmap("nanoGPT/data/shakespeare_char/train.bin", dtype=np.uint16, mode="r")
batch_size = 32 
block_size = 128
calibration_batches = 128  # 与校准时相同的批次数
generator = torch.Generator().manual_seed(1337)
starts_all = torch.randint(len(train_data) - block_size, (calibration_batches, batch_size), generator=generator)

with torch.no_grad():  # 只观察激活，不计算梯度或更新权重
    for batch_starts in starts_all:
        starts = batch_starts.tolist()
        inputs = np.stack([train_data[i:i + block_size] for i in starts]).astype(np.int64)
        targets = np.stack([train_data[i + 1:i + block_size + 1] for i in starts]).astype(np.int64)
        x = torch.tensor(inputs, dtype=torch.long, device=device)
        y = torch.tensor(targets, dtype=torch.long, device=device)
        context["tokens"] = x
        context["starts"] = batch_starts
        model(x, y)

for handle in handles:
    handle.remove()

def percentile_upper(histogram, maximum, fraction):
    rank = math.ceil(histogram.sum().item() * fraction)
    cumulative = histogram.cumsum(dim=0)
    index = torch.searchsorted(cumulative, torch.tensor(rank, dtype=torch.int64)).item()
    return (index + 1) * maximum / hist_bins

layer_results = {}

for name, record in accumulators.items():
    count = record["count"]
    assert count > 0
    assert histograms[name].sum().item() == count
    mean = record["sum"] / count
    mean_square = record["sum_sq"] / count
    p99 = percentile_upper(histograms[name], hist_limits[name], 0.99)
    p99_9 = percentile_upper(histograms[name], hist_limits[name], 0.999)
    p99_99 = percentile_upper(histograms[name], hist_limits[name], 0.9999)
    max_abs = max(abs(record["min"]), abs(record["max"]))
    layer_results[name] = {
        "count": count,  # 保存参与统计的激活元素个数
        "min": record["min"],  # 保存最小激活
        "max": record["max"],  # 保存最大激活
        "max_abs": max_abs,  # 保存最大绝对值
        "mean": mean,  # 保存带符号的均值
        "std": math.sqrt(max(0.0, mean_square - mean * mean)),  # 保存总体标准差
        "rms": math.sqrt(mean_square),  # 保存均方根
        "p99_abs_upper": p99,  # 保存绝对值 P99 所在区间的上边界
        "p99_9_abs_upper": p99_9,  # 保存绝对值 P99.9 所在区间的上边界
        "p99_99_abs_upper": p99_99,  # 保存绝对值 P99.99 所在区间的上边界
        "max_to_p99_9_ratio": max_abs / p99_9 if p99_9 > 0 else None,  # 衡量最大值比常见的大值突出多少
        "channel_max_abs": channel_maxima[name].tolist(),  # 保存每个输入通道的最大绝对值
        "mean_abs_quant_error": record["error_sum"] / count,  # 保存平均绝对量化误差
        "max_abs_quant_error": record["error_max"],  # 保存最大绝对量化误差
        "scale": calibration[name]["scale"],  # 保存本层实际使用的激活 scale
        "zero_point": calibration[name]["zero_point"],  # 保存本层实际使用的激活零点
        "top_events": sorted(top_events[name], key=lambda event: event["abs_value"], reverse=True)[:5],  # 保存 5 个最大的批次异常位置
        "abs_histogram_max": hist_limits[name],  # 保存直方图覆盖的绝对值上限
        "abs_histogram_counts": histograms[name].tolist(),  # 保存直方图各区间的全量计数
    }

result = {
    "source_checkpoint": "nanoGPT/out/task0-bf16/best_ckpt.pt",  # 记录使用的模型
    "calibration_batches": calibration_batches,  # 记录采集批数
    "batch_size": batch_size,  # 记录每批段数
    "block_size": block_size,  # 记录每段长度
    "seed": 1337,  # 记录随机种子
    "percentile_method": "4096-bin absolute-value histogram; report upper bin edge",  # 说明百分位是直方图区间上边界
    "layers": layer_results,  # 保存全部 17 层的统计
}

result_path = "task1/results/activation_stats.json"  # 指定结果文件位置
with open(result_path, "w", encoding="utf-8") as file:
    json.dump(result, file, indent=2)

example = layer_results["transformer.h.0.mlp.c_fc"]  # 取第一层 MLP 作为输出示例
print("recorded Linear layers:", len(layer_results))  # 应输出 17
print("first MLP P99.9 abs:", example["p99_9_abs_upper"])  # 查看该层绝对值 P99.9
print("first MLP max abs:", example["max_abs"])  # 查看该层最大绝对值
print("first MLP mean quant error:", example["mean_abs_quant_error"])  # 查看该层平均量化误差
print("saved to:", result_path)