import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

class Head(nn.Module):
    def __init__(self, in_channels, out, use_bn=True, act="silu"):
        super(Head, self).__init__()
        self.conv1 = conv3x3(in_channels, in_channels, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(in_channels) if use_bn else nn.Identity()
        if act == "silu":
            self.act1 = nn.SiLU(inplace=True)
            self.act2 = nn.SiLU(inplace=True)
        elif act == "relu":
            self.act1 = nn.ReLU(inplace=True)
            self.act2 = nn.ReLU(inplace=True)
        elif act in ("none", None):
            self.act1 = nn.Identity()
            self.act2 = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {act!r}")

        self.conv2 = conv3x3(in_channels, in_channels, bias=not use_bn)
        self.bn2 = nn.BatchNorm2d(in_channels) if use_bn else nn.Identity()
        self.head = nn.Conv2d(in_channels, out, kernel_size=1)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        head = self.head(x)

        return head

class Header(nn.Module):
    def __init__(self, num_classes, in_channels, use_bn=True, act="silu"):
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

