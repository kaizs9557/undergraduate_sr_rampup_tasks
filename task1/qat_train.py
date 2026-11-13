import sys
import json
import torch

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

# 4. 用一小批随机字符编号检查前向和反向传播
torch.manual_seed(1337)
model.train()

tokens = torch.randint(
    0, model.config.vocab_size, (2, 9), device=device
)
x = tokens[:, :-1]
y = tokens[:, 1:].contiguous()

_, loss = model(x, y)
loss.backward()

grad = model.transformer.h[0].mlp.c_fc.weight.grad
assert grad is not None

print("test loss:", loss.item())
print("mean absolute weight gradient:", grad.abs().mean().item())

model.zero_grad(set_to_none=True)