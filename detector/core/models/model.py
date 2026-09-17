import torch.nn as nn

from core.models.backbones.rpn import RPN
from core.models.backbones.mobilepixor import MobilePixorBackBone
from core.models.backbones.mobilepixor_coordinate_attention import (
    MobilePixorBackBone as MobilePixorCoordAttBackBone,
)
from core.models.backbones.pixor import PixorBackBone
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
        elif cfg["backbone"] == "pixor":
            self.backbone = PixorBackBone()
        elif cfg["backbone"] == "rpn":
            self.backbone = RPN()
        else:
            raise ValueError(f"Unsupported backbone: {cfg['backbone']!r}")

        self.num_classes = num_classes
        if cfg["cls_encoding"] == "binary":
            self.num_classes += 1

        self.header = Header(
            self.num_classes,
            cfg["backbone_out_dim"],
            box_encoding=cfg.get("box_encoding", "bev"),
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
