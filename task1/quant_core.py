import torch

#从quant_demo.py复制过来的函数，便于后续使用

#对称均匀量化
def quantize_symmetric(x):
    max_abs = x.abs().max()

    if max_abs.item() == 0:
        scale = x.new_tensor(1.0)
    else:
        scale = max_abs / 127

    q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return q, scale


def dequantize_symmetric(q, scale):
    x_hat = q.to(torch.float32) * scale
    return x_hat


#用外部数据做量化
def quantize_with_params(x, scale, zero_point):
    if scale <= 0:  # scale 必须大于 0，否则无法做除法
        raise ValueError("scale must be positive")
    q = torch.clamp(torch.round(x / scale) + zero_point, -128, 127).to(torch.int8)
    return q

def dequantize_with_params(q, scale, zero_point):
    return (q.to(torch.float32) - zero_point) * scale

def fake_quantize_with_params(x, scale, zero_point):
    q = quantize_with_params(x, scale, zero_point)
    return dequantize_with_params(q, scale, zero_point)

#非对称均匀量化
def quantize_asymmetric(x):
    x_min = min(x.min().item(), 0.0)
    x_max = max(x.max().item(), 0.0)

    if x_max == x_min:
        scale = 1.0
        zero_point = 0
    else:
        scale = (x_max - x_min) / 255
        zero_point = round(-128 - x_min / scale)
        zero_point = max(-128, min(127, zero_point))

    q = torch.clamp(
        torch.round(x / scale) + zero_point, -128, 127
    ).to(torch.int8)

    return q, scale, zero_point


def dequantize_asymmetric(q, scale, zero_point):
    return (q.to(torch.float32) - zero_point) * scale

#per-channel 权重量化
def quantize_weight_per_channel(w):
    max_abs = w.abs().amax(dim=1, keepdim=True)

    scale = torch.where(
        max_abs == 0,
        torch.ones_like(max_abs),
        max_abs / 127,
    )

    q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    return q, scale

def dequantize_weight_per_channel(q, scale):
    return q.to(torch.float32) * scale


def fake_quantize_weight_per_channel(w):
    q, scale = quantize_weight_per_channel(w)
    return dequantize_weight_per_channel(q, scale)


#数值参与前向计算，但不记录它的梯度路径
def fake_quantize_weight_per_channel_ste(w):
    q, scale = quantize_weight_per_channel(w.detach())
    w_hat = dequantize_weight_per_channel(q, scale)
    return w + (w_hat - w).detach()

def fake_quantize_with_params_ste(x, scale, zero_point):
    x_hat = fake_quantize_with_params(x.detach(), scale, zero_point)
    return x + (x_hat - x).detach()