"""Structural Reparameterization Building Blocks for MobilePixorNeXt.

Decouples multi-branch depthwise training (7x7 DW + 3x3 DW + Identity)
into a single equivalent 7x7 depthwise convolution for zero-latency inference.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def trans_conv_bn_to_kernel_bias(
    conv: nn.Conv2d,
    bn: nn.BatchNorm2d,
) -> Tuple[Tensor, Tensor]:
    """Fuse a Conv2d and BatchNorm2d pair into equivalent (kernel, bias)."""
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    std = torch.sqrt(var + eps)
    t = (gamma / std).reshape(-1, 1, 1, 1)

    kernel = conv.weight * t
    if conv.bias is not None:
        bias = beta + (conv.bias - mean) * (gamma / std)
    else:
        bias = beta - mean * (gamma / std)
    return kernel, bias


def trans_identity_bn_to_kernel_bias(
    bn: nn.BatchNorm2d,
    channels: int,
    kernel_size: int = 7,
) -> Tuple[Tensor, Tensor]:
    """Fuse an Identity BatchNorm2d into equivalent depthwise (kernel, bias)."""
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    std = torch.sqrt(var + eps)

    # Depthwise identity kernel (groups=channels) has shape (channels, 1, K, K)
    kernel = torch.zeros((channels, 1, kernel_size, kernel_size), device=bn.weight.device, dtype=bn.weight.dtype)
    center = kernel_size // 2
    for c in range(channels):
        kernel[c, 0, center, center] = 1.0

    t = (gamma / std).reshape(-1, 1, 1, 1)
    kernel = kernel * t
    bias = beta - mean * (gamma / std)
    return kernel, bias


class RepConv7x7(nn.Module):
    """Structural Reparameterization 7x7 Depthwise Convolution.

    Training Topology:
        Branch 1: 7x7 Depthwise Conv + BatchNorm
        Branch 2: 3x3 Depthwise Conv + BatchNorm
        Branch 3: Identity + BatchNorm
        Output = Branch1(X) + Branch2(X) + Branch3(X)

    Inference Topology (after switch_to_deploy()):
        Single 7x7 Depthwise Conv with bias (groups=channels).
    """

    def __init__(self, channels: int, deploy: bool = False):
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be positive, got {channels}")

        self.channels = channels
        self.deploy = deploy

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                channels,
                channels,
                kernel_size=7,
                stride=1,
                padding=3,
                groups=channels,
                bias=True,
            )
        else:
            self.rbr_conv7 = nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=7,
                    stride=1,
                    padding=3,
                    groups=channels,
                    bias=False,
                ),
                nn.BatchNorm2d(channels),
            )
            self.rbr_conv3 = nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=channels,
                    bias=False,
                ),
                nn.BatchNorm2d(channels),
            )
            self.rbr_identity = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        if self.deploy:
            return self.rbr_reparam(x)
        return self.rbr_conv7(x) + self.rbr_conv3(x) + self.rbr_identity(x)

    def get_equivalent_kernel_bias(self) -> Tuple[Tensor, Tensor]:
        """Compute the equivalent single 7x7 DW kernel and bias from the 3 branches."""
        assert not self.deploy, "Already deployed; cannot get equivalent kernel."

        kernel7, bias7 = trans_conv_bn_to_kernel_bias(self.rbr_conv7[0], self.rbr_conv7[1])
        kernel3, bias3 = trans_conv_bn_to_kernel_bias(self.rbr_conv3[0], self.rbr_conv3[1])
        kernel_id, bias_id = trans_identity_bn_to_kernel_bias(self.rbr_identity, self.channels, kernel_size=7)

        # Pad 3x3 kernel by 2 on all borders: (left, right, top, bottom)
        kernel3_padded = F.pad(kernel3, (2, 2, 2, 2))

        kernel_fused = kernel7 + kernel3_padded + kernel_id
        bias_fused = bias7 + bias3 + bias_id
        return kernel_fused, bias_fused

    def switch_to_deploy(self):
        """Fuse all branches into a single 7x7 Conv2d and delete training branches."""
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=7,
            stride=1,
            padding=3,
            groups=self.channels,
            bias=True,
        )
        self.rbr_reparam.weight.data.copy_(kernel)
        self.rbr_reparam.bias.data.copy_(bias)

        self.__delattr__("rbr_conv7")
        self.__delattr__("rbr_conv3")
        self.__delattr__("rbr_identity")
        self.deploy = True
