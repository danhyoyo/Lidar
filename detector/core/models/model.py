import torch
import torch.nn as nn

try:
    from core.bev_encoding import resolve_input_channels
    from core.detection_config import resolve_model_detection
    from core.models.backbones.registry import build_backbone
    from core.models.heads.cnn import Header
except ImportError:
    from detector.core.bev_encoding import resolve_input_channels
    from detector.core.detection_config import resolve_model_detection
    from detector.core.models.backbones.registry import build_backbone
    from detector.core.models.heads.cnn import Header


class CustomModel(nn.Module):
    def __init__(self, cfg, num_classes=4, input_channels=35, task_groups=None, *, box_mode="bev"):
        super(CustomModel, self).__init__()
        contract = resolve_model_detection(cfg, num_classes, task_groups, box_mode=box_mode)
        self.head_mode = contract.head_mode
        self.box_mode = contract.box_mode
        self.task_groups = contract.groups
        # A geometry-free legacy constructor keeps its explicit input width.
        # Pipeline construction supplies the authoritative data encoding/geometry.
        input_channels = resolve_input_channels(
            cfg.get("bev_encoding"),
            cfg.get("geometry") if "bev_encoding" in cfg else None,
            default_channels=input_channels,
        )

        backbone_name = str(cfg.get("backbone", "mobilepixor"))
        self.backbone = build_backbone(backbone_name, cfg, input_channels=input_channels)

        self.num_classes = num_classes
        cls_encoding = contract.cls_encoding
        if cls_encoding == "binary" and self.head_mode == "legacy_single":
            self.num_classes += 1

        is_mobilepixornext = backbone_name.lower() == "mobilepixornext"
        use_bn = cfg.get("header_use_bn", is_mobilepixornext)
        act = cfg.get("header_act", "silu" if is_mobilepixornext else "none")

        backbone_out_dim = cfg.get("backbone_out_dim", 16)

        use_iou = contract.use_iou
        if self.head_mode == "legacy_single":
            self.header = Header(
                self.num_classes, backbone_out_dim,
                use_bn=use_bn, act=act, use_iou=use_iou, box_mode=self.box_mode,
            )
        else:
            # Optional grouped topology stays out of legacy construction/imports.
            from .heads.grouped import GroupedHeader
            self.grouped_header = GroupedHeader(
                self.task_groups, backbone_out_dim, cls_encoding=cls_encoding,
                use_bn=use_bn, act=act, use_iou=use_iou, box_mode=self.box_mode,
            )

    def forward(self, x):
        if isinstance(x, dict) and "voxel" in x:
            x = x["voxel"]
        features = self.backbone(x)
        pred = (self.grouped_header(features) if self.head_mode == "grouped"
                else self.header(features))
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
