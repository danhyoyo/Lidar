import torch.nn as nn

from core.models.backbones.rpn import RPN
from core.models.backbones.mobilepixor import MobilePixorBackBone
from core.models.backbones.mobilepixor_coordinate_attention import (
    MobilePixorBackBone as MobilePixorCoordAttBackBone,
)
from core.models.backbones.pixor import PixorBackBone
from core.models.backbones.bevnext import BEVNeXtBackbone
from core.models.heads.cnn import Header

class CustomModel(nn.Module):
    def __init__(self, cfg, num_classes=4, input_channels=35):
        super(CustomModel, self).__init__()
        if cfg["backbone"] == "mobilepixor":
            self.backbone = MobilePixorBackBone()
        elif cfg["backbone"] == "mobilepixor_coordatt":
            self.backbone = MobilePixorCoordAttBackBone(
                input_channels=input_channels,
                scale_gated_fpn=cfg.get("scale_gated_fpn", False),
            )
        elif cfg["backbone"] == "bevnext":
            self.backbone = BEVNeXtBackbone(
                input_channels=input_channels,
                backbone_out_dim=cfg.get("backbone_out_dim", 16),
                c4_attention=cfg.get("c4_attention", "litemla"),
                scale_gated_fpn=cfg.get("scale_gated_fpn", True),
                expansion=cfg.get("expansion", 2.5),
            )
        elif cfg["backbone"] == "pixor":
            self.backbone = PixorBackBone()
        elif cfg["backbone"] == "rpn":
            self.backbone = RPN()
        else:
            raise ValueError(f"Unsupported backbone: {cfg['backbone']!r}")

        self.num_classes = num_classes
        if cfg["cls_encoding"] == "binary":
            self.num_classes += 1

        is_bevnext = cfg.get("backbone") == "bevnext"
        use_bn = cfg.get("header_use_bn", is_bevnext)
        act = cfg.get("header_act", "silu" if is_bevnext else "none")

        self.header = Header(
            self.num_classes,
            cfg["backbone_out_dim"],
            use_bn=use_bn,
            act=act,
        )

    def forward(self, x):
        features = self.backbone(x)
        pred = self.header(features)

        return pred


if __name__ == "__main__":
    cfg = {
        "backbone": "mobilepixor"
    }

    model = CustomModel(cfg)
    for p in model.backbone.parameters():
        print(p)
    #print(model.backbone.parameters)
