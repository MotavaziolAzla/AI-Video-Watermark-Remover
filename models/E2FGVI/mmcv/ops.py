import math
import torch
import torch.nn as nn
import torchvision.ops as ops
from torch.nn.modules.utils import _pair

def modulated_deform_conv2d(
    input: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    deform_groups: int = 1
) -> torch.Tensor:
    """
    Lightweight drop-in replacement for mmcv.ops.modulated_deform_conv2d using torchvision.ops.deform_conv2d.
    """
    return ops.deform_conv2d(
        input=input,
        offset=offset,
        weight=weight,
        bias=bias,
        stride=_pair(stride),
        padding=_pair(padding),
        dilation=_pair(dilation),
        mask=mask
    )

class ModulatedDeformConv2d(nn.Module):
    """
    Lightweight drop-in replacement for mmcv.ops.ModulatedDeformConv2d using torchvision.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deform_groups=1,
        bias=True
    ):
        super(ModulatedDeformConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deform_groups = deform_groups
        self.with_bias = bias

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, offset, mask):
        return modulated_deform_conv2d(
            x, offset, mask, self.weight, self.bias,
            self.stride, self.padding, self.dilation,
            self.groups, self.deform_groups
        )
