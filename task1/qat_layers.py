import torch
import torch.nn.functional as F

from quant_core import (
    fake_quantize_weight_per_channel_ste,
    fake_quantize_with_params_ste,
)


class QATLinear(torch.nn.Linear):
    def __init__(self, source, act_scale, act_zero_point):
        super().__init__(
            source.in_features,
            source.out_features,
            bias=source.bias is not None,
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        self.weight = source.weight
        if source.bias is not None:
            self.bias = source.bias

        self.act_scale = act_scale
        self.act_zero_point = act_zero_point

    def forward(self, x):
        x_hat = fake_quantize_with_params_ste(
            x, self.act_scale, self.act_zero_point
        )
        w_hat = fake_quantize_weight_per_channel_ste(self.weight)
        return F.linear(x_hat, w_hat, self.bias)