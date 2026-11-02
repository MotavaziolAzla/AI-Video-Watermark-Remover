import torch
import torch.nn as nn

class ConvModule(nn.Module):
    """
    Lightweight drop-in replacement for mmcv.cnn.ConvModule.
    Matches state_dict naming (self.conv.weight, self.conv.bias).
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
        bias=True,
        norm_cfg=None,
        act_cfg=dict(type='ReLU'),
        **kwargs
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        if act_cfg is not None and isinstance(act_cfg, dict):
            act_type = act_cfg.get('type')
            if act_type == 'ReLU':
                self.act = nn.ReLU(inplace=True)
            elif act_type == 'LeakyReLU':
                self.act = nn.LeakyReLU(negative_slope=act_cfg.get('negative_slope', 0.2), inplace=True)
            else:
                self.act = None
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
