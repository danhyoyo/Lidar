import torch.nn as nn

from core.models.backbones.registry import build_backbone
from core.models.heads.cnn import Header

class CustomModel(nn.Module):
    def __init__(self, cfg, num_classes=4, input_channels=35):
        super(CustomModel, self).__init__()
        self.backbone = build_backbone(cfg["backbone"], cfg, input_channels=input_channels)

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
