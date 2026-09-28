import torch.nn as nn

from core.models.backbones.registry import build_backbone
from core.models.heads.cnn import Header


class CustomModel(nn.Module):
    def __init__(self, cfg, num_classes=4, input_channels=35):
        super(CustomModel, self).__init__()
        backbone_name = str(cfg.get("backbone", "mobilepixor"))
        self.backbone = build_backbone(backbone_name, cfg, input_channels=input_channels)

        self.num_classes = num_classes
        cls_encoding = str(cfg.get("cls_encoding", "gaussian")).lower()
        if cls_encoding == "binary":
            self.num_classes += 1

        is_mobilepixornext = backbone_name.lower() == "mobilepixornext"
        use_bn = cfg.get("header_use_bn", is_mobilepixornext)
        act = cfg.get("header_act", "silu" if is_mobilepixornext else "none")

        backbone_out_dim = cfg.get("backbone_out_dim", 16)

        self.header = Header(
            self.num_classes,
            backbone_out_dim,
            use_bn=use_bn,
            act=act,
        )

    def forward(self, x):
        features = self.backbone(x)
        pred = self.header(features)
        return pred

    def switch_to_deploy(self):
        """Deploy-time weight fusion for the backbone."""
        if hasattr(self.backbone, "switch_to_deploy"):
            self.backbone.switch_to_deploy()
        return self

    def export_deploy_state_dict(self):
        """Switch to deploy mode and return the fused state_dict."""
        self.switch_to_deploy()
        return self.state_dict()



if __name__ == "__main__":
    cfg = {
        "backbone": "mobilepixor",
        "backbone_out_dim": 16,
        "cls_encoding": "gaussian",
    }

    model = CustomModel(cfg)
    print("CustomModel initialized successfully with default parameters:")
    print(f"Backbone: {type(model.backbone).__name__}")
    print(f"Header: {type(model.header).__name__} with heads: cls, offset, size, yaw")
