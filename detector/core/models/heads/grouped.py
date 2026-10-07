"""One independent existing BEV Header per resolved task group."""

import torch.nn as nn

from ...task_groups import validate_resolved_groups
from ...detection_config import resolve_box_mode
from .cnn import Header


class GroupedHeader(nn.Module):
    def __init__(self, task_groups, in_channels, cls_encoding="gaussian",
                 use_bn=False, act="none", use_iou=False, box_mode="bev"):
        super().__init__()
        self.box_mode = resolve_box_mode(box_mode)
        self.task_groups = validate_resolved_groups(task_groups)
        if not isinstance(cls_encoding, str) or cls_encoding.lower() not in ("gaussian", "binary"):
            raise ValueError("cls_encoding must be gaussian or binary")
        self.cls_encoding = cls_encoding.lower()
        if type(in_channels) is not int or in_channels <= 0:
            raise ValueError("in_channels must be a positive integer")
        if type(use_iou) is not bool or type(use_bn) is not bool:
            raise ValueError("use_iou and use_bn must be booleans")
        if any(hasattr(nn.ModuleDict, g.name) for g in self.task_groups):
            raise ValueError("Group name conflicts with installed ModuleDict API")
        self.heads = nn.ModuleDict({
            g.name: Header(g.num_classes + int(self.cls_encoding == "binary"),
                           in_channels, use_bn=use_bn, act=act, use_iou=use_iou,
                           box_mode=self.box_mode)
            for g in self.task_groups
        })

    def forward(self, features):
        return {"groups": {name: head(features) for name, head in self.heads.items()}}
