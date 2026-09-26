import torch.nn as nn

from core.models.backbones.rpn import RPN
from core.models.backbones.mobilepixor import MobilePixorBackBone
from core.models.backbones.pixor import PixorBackBone
from core.models.heads.cnn import Header

class CustomModel(nn.Module):
    def __init__(self, cfg, num_classes=4, input_channels=35):
        super(CustomModel, self).__init__()
        if cfg["backbone"] != "mobilepixor" and cfg.get("c4_attention", "none") != "none":
            raise ValueError("c4_attention is supported only by the mobilepixor backbone")
        if cfg["backbone"] == "mobilepixor":
            c2psa = cfg.get("c2psa", {})
            self.backbone = MobilePixorBackBone(
                input_channels=input_channels,
                c5_attention=cfg.get("c5_attention", "none"),
                c2psa_repeats=c2psa.get("repeats", 1),
                c2psa_expansion=c2psa.get("expansion", 0.5),
                c2psa_attn_ratio=c2psa.get("attn_ratio", 0.5),
                scale_gated_fpn=cfg.get("scale_gated_fpn", False),
                c4_attention=cfg.get("c4_attention", "none"),
                c4_attention_route=cfg.get("c4_attention_route", "shared"),
                lsk=cfg.get("lsk", {}),
                litemla=cfg.get("litemla", {}),
                dat=cfg.get("dat", {}),
                bra=cfg.get("bra", {}),
            )
        elif cfg["backbone"] == "mobilepixor_coordatt":
            from core.models.backbones.mobilepixor_coordinate_attention import (
                MobilePixorBackBone as MobilePixorCoordAttBackBone,
            )

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
            use_bn=cfg.get("header_use_bn", False),
            act=cfg.get("header_act", "none"),
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
