import torch
import torch.nn as nn

from core.models.backbones.rpn import RPN
from core.models.backbones.mobilepixor import MobilePixorBackBone
from core.models.backbones.mobilepixor_coordinate_attention import (
    MobilePixorBackBone as MobilePixorCoordAttBackBone,
)
from core.models.backbones.pixor import PixorBackBone
from core.models.heads.cnn import Header
from core.models.gaussian_visibility import GaussianPillarPropagation

class CustomModel(nn.Module):
    def __init__(self, cfg, num_classes=4, input_channels=35):
        super(CustomModel, self).__init__()
        gaussian = cfg.get("gaussian_propagation", {})
        mode = gaussian.get("mode", "none")
        if mode != "none" and cfg["backbone"] not in {"mobilepixor", "mobilepixor_coordatt"}:
            raise ValueError("Gaussian propagation requires a MobilePIXOR backbone")
        self.gaussian_propagation = (
            GaussianPillarPropagation(
                mode=mode,
                sigma_cells=float(gaussian.get("sigma_cells", 1.5)),
                radius_cells=int(gaussian.get("radius_cells", 4)),
                strength=float(gaussian.get("strength", 1.0)),
                kernel_type=gaussian.get("kernel_type", "gaussian"),
            ) if mode != "none" else None
        )
        backbone_input_channels = 8 if self.gaussian_propagation else input_channels
        if cfg["backbone"] == "mobilepixor":
            self.backbone = MobilePixorBackBone(input_channels=backbone_input_channels)
        elif cfg["backbone"] == "mobilepixor_coordatt":
            self.backbone = MobilePixorCoordAttBackBone(
                input_channels=backbone_input_channels,
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
        if cfg["cls_encoding"] == "gaussian":
            nn.init.constant_(self.header.cls.head.bias, -2.19)

    def forward(self, x):
        if self.gaussian_propagation is not None:
            x = self.gaussian_propagation(x)
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
