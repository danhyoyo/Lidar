import torch
from torch import nn

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

class Head(nn.Module):
    def __init__(self, in_channels, out):
        super(Head, self).__init__()
        self.conv1 = conv3x3(in_channels, in_channels)
        self.conv2 = conv3x3(in_channels, in_channels)
        self.head = nn.Conv2d(in_channels, out, kernel_size=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        head = self.head(x)

        return head

class Header(nn.Module):
    def __init__(self, num_classes, in_channels, box_encoding="bev",
                 predict_log_variance=False, initial_log_variance=-2.0):
        super(Header, self).__init__()
        if box_encoding not in {"bev", "center3d"}:
            raise ValueError(f"Unsupported box encoding: {box_encoding!r}")
        regression_channels = 3 if box_encoding == "center3d" else 2
        if predict_log_variance and box_encoding != "center3d":
            raise ValueError("predict_log_variance requires box_encoding='center3d'")
        self.predict_log_variance = predict_log_variance

        self.cls = Head(in_channels, num_classes)
        output_channels = regression_channels * (2 if predict_log_variance else 1)
        self.offset = Head(in_channels, output_channels)
        self.size = Head(in_channels, output_channels)
        self.yaw = Head(in_channels, 2)
        if predict_log_variance:
            for head in (self.offset, self.size):
                nn.init.zeros_(head.head.weight[regression_channels:])
                nn.init.constant_(head.head.bias[regression_channels:], initial_log_variance)


    def forward(self, x):
        cls = self.cls(x)
        offset = self.offset(x)
        size = self.size(x)
        yaw = self.yaw(x)

        pred = {"cls": cls, "offset": offset, "size": size, "yaw": yaw}
        if self.predict_log_variance:
            offset, offset_log_var = offset.chunk(2, dim=1)
            size, size_log_var = size.chunk(2, dim=1)
            pred.update(offset=offset, size=size,
                        log_var=torch.cat((offset_log_var, size_log_var), dim=1))

        return pred
