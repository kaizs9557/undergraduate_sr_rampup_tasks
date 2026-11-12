import sys
import torch

sys.path.insert(0, "nanoGPT")
from model import GPT, GPTConfig

checkpoint = torch.load(
    "nanoGPT/out/task0-bf16/best_ckpt.pt",
    map_location="cpu",
    weights_only=False,
)

config = GPTConfig(**checkpoint["model_args"])
model = GPT(config)
model.load_state_dict(checkpoint["model"])
model.eval()

x = torch.zeros((1, 8), dtype=torch.long)

with torch.no_grad():
    logits, loss = model(x)

print("checkpoint step:", checkpoint["iter_num"])
print("best validation loss:", checkpoint["best_val_loss"])
print("logits shape:", logits.shape)
print("loss:", loss)

linear_count = 0  # 记录找到了多少个 Linear 层
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        weight = module.weight.detach()
        print(name, tuple(weight.shape))  # 打印层名，以及权重矩阵的形状
        linear_count += 1
print("total Linear layers:", linear_count)  # 打印 Linear 层总数