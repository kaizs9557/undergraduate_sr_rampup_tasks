import sys
import math
import numpy as np
import torch
import json

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

mean_loss = sum(losses) / len(losses) #100个batch平均loss
ppl = math.exp(mean_loss)

print("device:", device)
print("validation tokens:", eval_batches * batch_size * block_size)
print("validation loss:", mean_loss)
print("validation PPL:", ppl)

metrics = {
    "run_tag": "baseline_fp32",
    "checkpoint": "nanoGPT/out/task0-bf16/best_ckpt.pt",
    "evaluation_precision": "fp32",
    "eval_seed": 1337,
    "eval_batches": eval_batches,
    "batch_size": batch_size,
    "block_size": block_size,
    "validation_tokens": eval_batches * batch_size * block_size,
    "validation_loss": mean_loss,
    "validation_ppl": ppl,
}

result_path = "task1/results/baseline_metrics.json"
with open(result_path, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)
print("metrics saved to:", result_path)