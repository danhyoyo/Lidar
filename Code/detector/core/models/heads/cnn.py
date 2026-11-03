import torch.nn as nn
from ...detection_config import resolve_box_mode, resolve_header_architecture

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

class Head(nn.Module):
    def __init__(self, in_channels, out, use_bn=False, act="none", *, num_convs=2):
        super(Head, self).__init__()
        if type(num_convs) is not int or num_convs not in (1, 2):
            raise ValueError("num_convs must be 1 or 2")
        self.conv1 = conv3x3(in_channels, in_channels)
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

        if num_convs == 2:
            self.conv2 = conv3x3(in_channels, in_channels)
            self.bn2 = nn.BatchNorm2d(in_channels) if use_bn else nn.Identity()
        else:
            self.conv2 = nn.Identity()
            self.bn2 = nn.Identity()
            self.act2 = nn.Identity()
        self.head = nn.Conv2d(in_channels, out, kernel_size=1)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        head = self.head(x)

        return head

class Header(nn.Module):
    def __init__(self, num_classes, in_channels, use_bn=False, act="none", use_iou=False,
                 box_mode="bev", *, header_architecture="independent"):
        super(Header, self).__init__()
        self.box_mode = resolve_box_mode(box_mode)
        self.header_architecture = resolve_header_architecture(header_architecture)
        self.use_iou = use_iou
        num_convs = 2
        if self.header_architecture == "shared":
            # Each grouped Header owns this block. It replaces the first
            # private convolution; every prediction path still has two 3x3s.
            if act == "silu":
                shared_act = nn.SiLU(inplace=True)
            elif act == "relu":
                shared_act = nn.ReLU(inplace=True)
            elif act in ("none", None):
                shared_act = nn.Identity()
            else:
                raise ValueError(f"Unsupported activation: {act!r}")
            self.shared = nn.Sequential(
                conv3x3(in_channels, in_channels),
                nn.BatchNorm2d(in_channels) if use_bn else nn.Identity(),
                shared_act,
            )
            num_convs = 1
        # Preserve the historical construction order, names and RNG sequence
        # when independent is selected or the new option is omitted.
        options = dict(use_bn=use_bn, act=act, num_convs=num_convs)
        self.cls = Head(in_channels, num_classes, **options)
        self.offset = Head(in_channels, 2, **options)
        self.size = Head(in_channels, 2, **options)
        self.yaw = Head(in_channels, 2, **options)
        if self.use_iou:
            self.iou = Head(in_channels, 1, **options)
        if self.box_mode == "3d":
            # Bottom-center z and log(height); BEV branch widths stay unchanged.
            self.vertical = Head(in_channels, 2, **options)

    def forward(self, x):
        if self.header_architecture == "shared":
            x = self.shared(x)
        cls = self.cls(x)
        offset = self.offset(x)
        size = self.size(x)
        yaw = self.yaw(x)

        pred = {"cls": cls, "offset": offset, "size": size, "yaw": yaw}
        if self.use_iou:
            pred["iou"] = self.iou(x)
        if self.box_mode == "3d":
            pred["vertical"] = self.vertical(x)

        return pred
