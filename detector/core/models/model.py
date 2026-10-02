import torch
import torch.nn as nn

try:
    from detector.core.models.backbones.registry import build_backbone
    from detector.core.models.heads.cnn import Header
    from detector.core.models.encoders.rich_mamba import RichMambaEncoder
except ImportError:
    from core.models.backbones.registry import build_backbone
    from core.models.heads.cnn import Header
    from core.models.encoders.rich_mamba import RichMambaEncoder


class CustomModel(nn.Module):
    def __init__(self, cfg, num_classes=4, input_channels=35):
        super(CustomModel, self).__init__()
        bev_cfg = cfg.get("bev_encoding", {})
        bev_name = bev_cfg.get("name", "binary_slices")
        geometry = cfg.get("geometry", {
            "x_min": 0.0, "x_max": 70.4, "x_res": 0.1,
            "y_min": -40.0, "y_max": 40.0, "y_res": 0.1,
            "z_min": -2.5, "z_max": 1.0, "z_res": 0.1,
        })

        if bev_name == "rich_mamba":
            self.encoder = RichMambaEncoder(bev_cfg, geometry)
            input_channels = int(bev_cfg.get("out_channels", 8))
        else:
            self.encoder = None

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

        use_iou = bool(cfg.get("header_use_iou", False))
        self.header = Header(
            self.num_classes,
            backbone_out_dim,
            use_bn=use_bn,
            act=act,
            use_iou=use_iou,
        )

    def forward(self, x):
        if self.encoder is not None:
            # Check if input is points rather than already a BEV 4D tensor (B, C, H, W)
            if isinstance(x, (list, tuple)) or (isinstance(x, torch.Tensor) and x.ndim <= 3):
                x = self.encoder(x)
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
