# Task 0 基线 A：严格 FP32
out_dir = 'out/task0-bf16'
run_tag = 'T0-BF16-001'
results_dir = '../task0/results'

wandb_log = True
wandb_project = 'rampup-task0'
wandb_run_name = 'T0-BF16-001'

dataset = 'shakespeare_char'

n_layer = 4
n_head = 4
n_embd = 128
block_size = 128
dropout = 0.1
bias = False

batch_size = 32
gradient_accumulation_steps = 1
max_iters = 3000
lr_decay_iters = 3000
warmup_iters = 100
learning_rate = 1e-3
min_lr = 1e-4
beta2 = 0.99
weight_decay = 1e-1
grad_clip = 1.0

eval_interval = 250
eval_iters = 100
log_interval = 10

dtype = 'bfloat16'
tf32 = False
compile = False
device = 'cuda'
init_from = 'scratch'

profile = False