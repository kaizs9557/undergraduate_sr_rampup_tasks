import torch

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


x = torch.tensor([[-1.0, 0.0], [0.25, 1.0]], dtype=torch.float32)

q, scale = quantize_symmetric(x)
x_hat = dequantize_symmetric(q, scale)
error = (x - x_hat).abs()

print("scale:", scale.item())
print("quantized:", q)
print("dequantized:", x_hat)
print("max absolute error:", error.max().item())

zero = torch.zeros((2, 2), dtype=torch.float32)
zero_q, zero_scale = quantize_symmetric(zero)
print("all-zero test:", zero_q, "scale:", zero_scale.item())


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


a = torch.tensor([[-1.0, 0.0], [1.0, 2.0]], dtype=torch.float32)
a_q, a_scale, a_zero_point = quantize_asymmetric(a)
a_hat = dequantize_asymmetric(a_q, a_scale, a_zero_point)

print("asymmetric scale:", a_scale)
print("asymmetric zero_point:", a_zero_point)
print("asymmetric q:", a_q)
print("asymmetric dequantized:", a_hat)


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


w = torch.tensor(
    [[-1.0, -0.25, 0.25, 1.0],
     [-0.01, -0.005, 0.005, 0.01]],
    dtype=torch.float32,
)

w_q, w_scale = quantize_weight_per_channel(w)
print("per-channel scales:", w_scale)
print("per-channel integers:", w_q)

def dequantize_weight_per_channel(q, scale):
    return q.to(torch.float32) * scale


def fake_quantize_weight_per_channel(w):
    q, scale = quantize_weight_per_channel(w)
    return dequantize_weight_per_channel(q, scale)


tensor_q, tensor_scale = quantize_symmetric(w)
tensor_hat = dequantize_symmetric(tensor_q, tensor_scale)
channel_hat = fake_quantize_weight_per_channel(w)

tensor_error = (w[1] - tensor_hat[1]).abs().max().item()
channel_error = (w[1] - channel_hat[1]).abs().max().item()

print("small row error, per-tensor:", tensor_error)
print("small row error, per-channel:", channel_error)
print("fake-quantized data type:", channel_hat.dtype)