import sys
import json
import torch
import numpy as np

sys.path.insert(0, "nanoGPT")
from model import GPT, GPTConfig
from qat_layers import QATLinear

# 1. 加载 Task 0 训练好的模型
checkpoint = torch.load(
    "nanoGPT/out/task0-bf16/best_ckpt.pt",
    map_location="cpu",
    weights_only=False,
)
model = GPT(GPTConfig(**checkpoint["model_args"]))
model.load_state_dict(checkpoint["model"])

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
torch.backends.cuda.matmul.allow_tf32 = False

# 2. 读取之前校准好的激活量化参数
with open("task1/results/activation_ranges.json", "r", encoding="utf-8") as file:
    all_params = json.load(file)

# 3. 把原来的 Linear 层逐个替换成 QATLinear
converted_names = []

for name, layer in list(model.named_modules()):
    if not isinstance(layer, torch.nn.Linear):
        continue

    params = all_params[name]
    parent_path, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model

    qat_layer = QATLinear(
        layer,
        params["scale"],
        params["zero_point"],
    )
    setattr(parent, child_name, qat_layer)
    converted_names.append(name)

print("QAT Linear layers:", len(converted_names))
assert len(converted_names) == 17

# 4. 设置 QAT 训练
torch.manual_seed(1337)
model.train()
train_data = np.memmap("nanoGPT/data/shakespeare_char/train.bin", dtype=np.uint16, mode="r")
batch_size = 32  # 每步取 32 段文本
block_size = 128  # 每段输入包含 128 个字符编号
train_steps = 400  # 本次简化 QAT 训练 400 步
learning_rate = 5e-5  # 继续训练时使用较小的学习率
train_generator = torch.Generator().manual_seed(1337)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), weight_decay=0.1)  # 创建更新模型权重的 AdamW 优化器
loss_history = []

def get_train_batch():
    starts = torch.randint(len(train_data) - block_size, (batch_size,), generator=train_generator).tolist()
    x_array = np.stack([train_data[i:i + block_size] for i in starts]).astype(np.int64)
    y_array = np.stack([train_data[i + 1:i + block_size + 1] for i in starts]).astype(np.int64)
    x = torch.tensor(x_array, dtype=torch.long, device=device)
    y = torch.tensor(y_array, dtype=torch.long, device=device)
    return x, y

# 5. 逐步训练
for step in range(1, train_steps + 1):
    x, y = get_train_batch()
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    if not torch.isfinite(loss).item():
        raise RuntimeError(f"non-finite loss at step {step}")
    loss.backward() 
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    loss_value = loss.item()
    loss_history.append(loss_value)
    if step == 1 or step % 20 == 0:  # 第一步和之后每 20 步打印一次
        print(f"QAT step {step}/{train_steps}, train loss: {loss_value:.4f}")  # 显示训练进度

# 6. 保存训练后的模型
qat_checkpoint_path = "task1/results/q3_qat_checkpoint.pt"
qat_checkpoint = {
    "model": model.state_dict(),
    "model_args": checkpoint["model_args"],
    "source_checkpoint": "nanoGPT/out/task0-bf16/best_ckpt.pt",
    "activation_ranges": "task1/results/activation_ranges.json",
    "train_steps": train_steps,
    "learning_rate": learning_rate,
    "seed": 1337,
}
torch.save(qat_checkpoint, qat_checkpoint_path)

# 7. 保存可阅读的训练记录
log_path = "task1/results/q3_train_log.json"
train_log = {
    "source_checkpoint": qat_checkpoint["source_checkpoint"],  # 记录起始模型
    "weight_quantization": "symmetric_int8_per_channel_ste",  # 记录权重量化方法
    "activation_quantization": "asymmetric_int8_per_tensor_static_ste",  # 记录激活量化方法
    "train_steps": train_steps,
    "batch_size": batch_size,
    "block_size": block_size,
    "learning_rate": learning_rate,
    "seed": 1337,
    "train_losses": loss_history,
}
with open(log_path, "w", encoding="utf-8") as file:
    json.dump(train_log, file, indent=2)
print("QAT checkpoint saved to:", qat_checkpoint_path)
print("QAT training log saved to:", log_path)