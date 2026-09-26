from torch import nn


SUPPORTED_HEAD_ACTIVATIONS = {"none", "relu", "silu"}

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

class Head(nn.Module):
    def __init__(self, in_channels, out, use_bn=False, act="none"):
        super(Head, self).__init__()
        if type(use_bn) is not bool:
            raise ValueError("use_bn must be a boolean")
        act = "none" if act is None else act
        if act not in SUPPORTED_HEAD_ACTIVATIONS:
            raise ValueError(
                f"Unsupported head activation {act!r}; expected one of "
                f"{sorted(SUPPORTED_HEAD_ACTIVATIONS)}"
            )

        # Keep convolution bias=False in every mode. This is the exact legacy
        # parameterization when BN/activation are disabled and avoids a
        # redundant bias when BatchNorm2d is enabled.
        self.conv1 = conv3x3(in_channels, in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels) if use_bn else nn.Identity()
        self.act1 = self._make_activation(act)
        self.conv2 = conv3x3(in_channels, in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels) if use_bn else nn.Identity()
        self.act2 = self._make_activation(act)
        self.head = nn.Conv2d(in_channels, out, kernel_size=1)

    @staticmethod
    def _make_activation(act):
        if act == "silu":
            return nn.SiLU(inplace=True)
        if act == "relu":
            return nn.ReLU(inplace=True)
        return nn.Identity()

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        head = self.head(x)

        return head

class Header(nn.Module):
    def __init__(self, num_classes, in_channels, use_bn=False, act="none"):
        super(Header, self).__init__()
        self.cls = Head(in_channels, num_classes, use_bn=use_bn, act=act)
        self.offset = Head(in_channels, 2, use_bn=use_bn, act=act)
        self.size = Head(in_channels, 2, use_bn=use_bn, act=act)
        self.yaw = Head(in_channels, 2, use_bn=use_bn, act=act)


    def forward(self, x):
        cls = self.cls(x)
        offset = self.offset(x)
        size = self.size(x)
        yaw = self.yaw(x)

        pred = {"cls": cls, "offset": offset, "size": size, "yaw": yaw}

        return pred
