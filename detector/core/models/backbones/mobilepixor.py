import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Optional, List
from core.models.backbones.c4_attention import build_c4_attention


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

def conv3x3_dw(in_planes, out_planes, stride = 1, bias = False):
    return nn.Sequential(
        # dw
        nn.Conv2d(in_planes, in_planes, 3, stride, 1, groups=in_planes, bias=bias),
        nn.BatchNorm2d(in_planes),
        nn.ReLU(inplace=True),

        # pw
        nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
        )



class ConvNormActivation(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
    ) -> None:

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.out_channels = out_channels


class Conv2dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer wont be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optinal): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            torch.nn.Conv2d,
        )


class InvertedResidual(nn.Module):
    def __init__(
        self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 insted of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU)
            )
        layers.extend(
            [
                # dw
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class C2PSAAttention(nn.Module):
    """Global position-sensitive attention used inside :class:`C2PSA`.

    This is a local, dependency-free implementation of the C2PSA attention
    equations used by modern Ultralytics YOLO models. It operates on a BEV
    feature map and preserves its channel count and spatial resolution.
    """

    def __init__(self, channels: int, num_heads: int = 1, attn_ratio: float = 0.5):
        super().__init__()
        if channels < 1 or num_heads < 1 or channels % num_heads:
            raise ValueError("channels must be positive and divisible by num_heads")
        if not 0.0 < attn_ratio <= 1.0:
            raise ValueError("attn_ratio must be in (0, 1]")

        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.key_dim = max(1, int(self.head_dim * attn_ratio))
        self.scale = self.key_dim ** -0.5
        key_channels = self.key_dim * num_heads

        self.qkv = Conv2dNormActivation(
            channels,
            channels + 2 * key_channels,
            kernel_size=1,
            activation_layer=None,
        )
        self.proj = Conv2dNormActivation(
            channels, channels, kernel_size=1, activation_layer=None
        )
        self.position = Conv2dNormActivation(
            channels,
            channels,
            kernel_size=3,
            groups=channels,
            activation_layer=None,
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        tokens = height * width
        qkv = self.qkv(x).reshape(
            batch,
            self.num_heads,
            2 * self.key_dim + self.head_dim,
            tokens,
        )
        query, key, value = qkv.split(
            (self.key_dim, self.key_dim, self.head_dim), dim=2
        )
        weights = torch.matmul(query.transpose(-2, -1), key) * self.scale
        weights = weights.softmax(dim=-1)
        attended = torch.matmul(value, weights.transpose(-2, -1)).reshape(
            batch, channels, height, width
        )
        value_map = value.reshape(batch, channels, height, width)
        return self.proj(attended + self.position(value_map))


class C2PSABlock(nn.Module):
    """One residual attention and point-wise feed-forward block."""

    def __init__(self, channels: int, num_heads: int, attn_ratio: float):
        super().__init__()
        self.attention = C2PSAAttention(channels, num_heads, attn_ratio)
        self.feed_forward = nn.Sequential(
            Conv2dNormActivation(
                channels,
                2 * channels,
                kernel_size=1,
                activation_layer=nn.SiLU,
            ),
            Conv2dNormActivation(
                2 * channels,
                channels,
                kernel_size=1,
                activation_layer=None,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(x)
        return x + self.feed_forward(x)


class C2PSA(nn.Module):
    """Cross-stage partial position-sensitive attention for C5 BEV features."""

    def __init__(
        self,
        channels: int,
        repeats: int = 1,
        expansion: float = 0.5,
        attn_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        if repeats < 1:
            raise ValueError("C2PSA repeats must be positive")
        if not 0.0 < expansion <= 1.0:
            raise ValueError("C2PSA expansion must be in (0, 1]")

        hidden_channels = int(channels * expansion)
        if hidden_channels < 1:
            raise ValueError("C2PSA expansion produced zero hidden channels")
        num_heads = max(hidden_channels // 64, 1)
        if hidden_channels % num_heads:
            raise ValueError("C2PSA hidden channels must be divisible by num_heads")

        self.hidden_channels = hidden_channels
        self.input_projection = Conv2dNormActivation(
            channels,
            2 * hidden_channels,
            kernel_size=1,
            activation_layer=nn.SiLU,
        )
        self.blocks = nn.Sequential(
            *(
                C2PSABlock(hidden_channels, num_heads, attn_ratio)
                for _ in range(repeats)
            )
        )
        self.output_projection = Conv2dNormActivation(
            2 * hidden_channels,
            channels,
            kernel_size=1,
            activation_layer=nn.SiLU,
        )

    def forward(self, x: Tensor) -> Tensor:
        bypass, attended = self.input_projection(x).split(
            (self.hidden_channels, self.hidden_channels), dim=1
        )
        attended = self.blocks(attended)
        return self.output_projection(torch.cat((bypass, attended), dim=1))


class MobilePixorBackBone(nn.Module):

    def __init__(
        self,
        block=InvertedResidual,
        use_bn=True,
        input_channels=35,
        c5_attention="none",
        c2psa_repeats=1,
        c2psa_expansion=0.5,
        c2psa_attn_ratio=0.5,
        scale_gated_fpn=False,
        c4_attention="none",
        c4_attention_route="shared",
        lsk=None,
        litemla=None,
    ):
        super(MobilePixorBackBone, self).__init__()

        self.use_bn = use_bn

        attention_name = str(c5_attention).lower()
        if attention_name == "none":
            self.c5_attention = nn.Identity()
        elif attention_name == "c2psa":
            self.c5_attention = C2PSA(
                96,
                repeats=int(c2psa_repeats),
                expansion=float(c2psa_expansion),
                attn_ratio=float(c2psa_attn_ratio),
            )
        else:
            raise ValueError(
                f"Unsupported C5 attention {c5_attention!r}; expected 'none' or 'c2psa'"
            )
        self.c5_attention_name = attention_name
        if not isinstance(scale_gated_fpn, bool):
            raise ValueError("scale_gated_fpn must be a boolean")
        self.scale_gated_fpn = scale_gated_fpn

        # Block 1
        self.conv1 = conv3x3(input_channels, 32)
        self.conv2 = conv3x3(32, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)


        # Block 2-5
        self.input_channel = 32
        self.block2 = self._make_layer(block, 1, 24, 1, 2)
        self.block3 = self._make_layer(block, 6, 32, 3, 2)
        self.block4 = self._make_layer(block, 6, 64, 4, 2)
        self.block5 = self._make_layer(block, 6, 96, 3, 2)

        # self.block3 = self._make_layer(block, 48, num_blocks=num_block[1])
        # self.block4 = self._make_layer(block, 64, num_blocks=num_block[2])
        # self.block5 = self._make_layer(block, 96, num_blocks=num_block[3])

        # Lateral layers
        self.latlayer1 = nn.Conv2d(96, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        p = 1
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1, p))

        # Optional scale-gated lateral fusion, ported from the deprecated
        # MobileBEV branch. Zero initialization makes each gate exactly equal
        # to the original sum-FPN at initialization: 2 * sigmoid(0) == 1.
        if self.scale_gated_fpn:
            self.gate_c4 = nn.Conv2d(
                32, 32, kernel_size=3, padding=1, groups=32, bias=True
            )
            self.gate_c3 = nn.Conv2d(
                16, 16, kernel_size=3, padding=1, groups=16, bias=True
            )
            nn.init.zeros_(self.gate_c4.weight)
            nn.init.zeros_(self.gate_c4.bias)
            nn.init.zeros_(self.gate_c3.weight)
            nn.init.zeros_(self.gate_c3.bias)

        # Construct last so a disabled adapter does not change existing state
        # keys or seeded initialization. Routing decides whether refined C4 is
        # shared with block5 or reserved for the lateral FPN branch.
        self.c4_attention = build_c4_attention(c4_attention, lsk=lsk, litemla=litemla)
        self.c4_attention_name = c4_attention.lower()
        route = str(c4_attention_route).lower()
        if route not in ("shared", "lateral_only"):
            raise ValueError(
                "c4_attention_route must be 'shared' or 'lateral_only'"
            )
        self.c4_attention_route = route

    def forward(self, x):
        #print("x.shape")
        #print(x.shape)
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        c1 = self.relu(x)

        # bottom up layers
        c2 = self.block2(c1)
        c3 = self.block3(c2)
        c4_raw = self.block4(c3)
        c4_refined = self.c4_attention(c4_raw)
        c5_input = (
            c4_refined if self.c4_attention_route == "shared" else c4_raw
        )
        c5 = self.block5(c5_input)
        c5 = self.c5_attention(c5)

        l5 = self.latlayer1(c5)
        l4 = self.latlayer2(c4_refined)
        u4 = self.deconv1(l5)
        p5 = (
            u4 + 2 * torch.sigmoid(self.gate_c4(l4 + u4)) * l4
            if self.scale_gated_fpn
            else l4 + u4
        )
        l3 = self.latlayer3(c3)
        u3 = self.deconv2(p5)
        p4 = (
            u3 + 2 * torch.sigmoid(self.gate_c3(l3 + u3)) * l3
            if self.scale_gated_fpn
            else l3 + u3
        )

        return p4


    def _make_layer(self, block, t, c, n, s):
        #output_channel = _make_divisible(c * width_mult, round_nearest)
        output_channel = c
        features = []
        for i in range(n):
            stride = s if i == 0 else 1
            features.append(block(self.input_channel, output_channel, stride, expand_ratio=t))
            self.input_channel = output_channel

        return nn.Sequential(*features)
