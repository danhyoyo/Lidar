"""Functional, numerical, and integration checks for the C4 ablations."""

from copy import deepcopy
import io
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/kitti_training_pipeline"))
from common import build_model, configure_detector_imports, read_json
from ablation import ablation_label, config_digest, resolve_ablation_config

configure_detector_imports(ROOT / "detector")
from core.models.backbones.c4_attention import (
    LSKRefinement, LiteMLARefinement, build_c4_attention,
)
from core.models.backbones.mobilepixor import MobilePixorBackBone
from core.losses.loss_fn import LossFunction

CONFIG_DIR = ROOT / "configs/kitti/backbone_branch"


def config_differences(actual, expected, path=""):
    """Return concise field-level differences for controlled preset failures."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        differences = []
        for key in sorted(actual.keys() | expected.keys()):
            child = f"{path}.{key}" if path else key
            if key not in actual:
                differences.append(f"{child}: missing; expected {expected[key]!r}")
            elif key not in expected:
                differences.append(f"{child}: unexpected {actual[key]!r}")
            else:
                differences.extend(config_differences(actual[key], expected[key], child))
        return differences
    if actual != expected:
        return [f"{path}: {actual!r} != {expected!r}"]
    return []


class C4AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(23)
        self.config = read_json(CONFIG_DIR / "kitti_mobilepixor_baseline.json")

    def test_disabled_matches_original_forward_and_checkpoint(self):
        # Legacy checkpoints have no c4_attention keys. Missing and explicit
        # 'none' must have identical seeded weights, outputs, and state keys.
        cfg = deepcopy(self.config)
        cfg["model"].pop("c4_attention")
        torch.manual_seed(5)
        original = build_model(cfg).eval()
        torch.manual_seed(5)
        explicit = build_model(self.config).eval()
        explicit.load_state_dict(original.state_dict(), strict=True)
        self.assertFalse(any("c4_attention" in k for k in explicit.state_dict()))
        self.assertEqual(sum(p.numel() for p in explicit.parameters()), 597817)
        x = torch.randn(1, 35, 32, 48)
        with torch.no_grad():
            actual = explicit(x)
            b = original.backbone
            c1 = b.relu(b.bn2(b.conv2(b.relu(b.bn1(b.conv1(x))))))
            c3 = b.block3(b.block2(c1))
            c4 = b.block4(c3)
            c5 = b.block5(c4)
            fused4 = b.latlayer2(c4) + b.deconv1(b.latlayer1(c5))
            features = b.latlayer3(c3) + b.deconv2(fused4)
            expected = original.header(features)
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_linear_attention_equals_explicit_normalized_kernel_attention(self):
        layer = LiteMLARefinement(channels=8, head_dim=4).double()
        packed = torch.randn(2, 48, 3, 5, dtype=torch.float64, requires_grad=True)
        actual = layer.linear_attention(packed)
        q, k, v = packed.reshape(2, -1, 12, 15).split(4, dim=2)
        weights = k.relu().transpose(-1, -2) @ q.relu()
        weights = weights / (weights.sum(dim=-2, keepdim=True) + layer.eps)
        expected = (v @ weights).reshape_as(actual)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        ga = torch.autograd.grad(actual.square().sum(), packed, retain_graph=True)[0]
        ge = torch.autograd.grad(expected.square().sum(), packed)[0]
        torch.testing.assert_close(ga, ge, rtol=1e-9, atol=1e-9)

    def test_zero_and_large_half_precision_attention_are_finite(self):
        layer = LiteMLARefinement(channels=8, head_dim=4)
        for dtype in (torch.float16, torch.bfloat16):
            for magnitude in (0.0, 10000.0):
                packed = torch.full((1, 24, 8, 12), magnitude, dtype=dtype, requires_grad=True)
                output = layer.linear_attention(packed)
                self.assertEqual(output.dtype, dtype)
                self.assertTrue(torch.isfinite(output).all())
                output.float().mean().backward()
                self.assertTrue(torch.isfinite(packed.grad).all())

    def test_residual_modules_preserve_shape_and_learn(self):
        for module_type in (LSKRefinement, LiteMLARefinement):
            with self.subTest(module=module_type.__name__):
                layer = module_type().train()
                sample = torch.randn(2, 64, 9, 11, requires_grad=True)
                output = layer(sample)
                self.assertEqual(output.shape, sample.shape)
                output.square().mean().backward()
                for name, parameter in layer.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(layer.layer_scale.grad.abs().sum().item(), 0)
                weight = layer.local.weight if isinstance(layer, LSKRefinement) else layer.qkv.weight
                self.assertGreater(weight.grad.abs().sum().item(), 0)
                with torch.no_grad():
                    layer.layer_scale.zero_()
                    torch.testing.assert_close(layer(sample), sample, rtol=0, atol=0)

    def test_shared_route_feeds_refined_c4_to_both_consumers(self):
        for mode in ("lsk", "litemla"):
            backbone = MobilePixorBackBone(c4_attention=mode).eval()
            captured = {}
            handles = [
                backbone.c4_attention.register_forward_hook(
                    lambda m, args, out: captured.update(refined=out)),
                backbone.block5.register_forward_pre_hook(
                    lambda m, args: captured.update(c5_input=args[0])),
                backbone.latlayer2.register_forward_pre_hook(
                    lambda m, args: captured.update(lateral_input=args[0])),
            ]
            with torch.no_grad():
                backbone(torch.randn(1, 35, 32, 48))
            for handle in handles:
                handle.remove()
            self.assertIs(captured["refined"], captured["c5_input"])
            self.assertIs(captured["refined"], captured["lateral_input"])

    def test_lateral_only_route_keeps_block5_on_raw_c4(self):
        backbone = MobilePixorBackBone(
            c4_attention="litemla", c4_attention_route="lateral_only"
        ).eval()
        captured = {}
        handles = [
            backbone.block4.register_forward_hook(
                lambda m, args, out: captured.update(raw=out)),
            backbone.c4_attention.register_forward_hook(
                lambda m, args, out: captured.update(refined=out)),
            backbone.block5.register_forward_pre_hook(
                lambda m, args: captured.update(c5_input=args[0])),
            backbone.latlayer2.register_forward_pre_hook(
                lambda m, args: captured.update(lateral_input=args[0])),
        ]
        with torch.no_grad():
            backbone(torch.randn(1, 35, 32, 48))
        for handle in handles:
            handle.remove()
        self.assertIs(captured["raw"], captured["c5_input"])
        self.assertIs(captured["refined"], captured["lateral_input"])
        self.assertIsNot(captured["raw"], captured["refined"])

    def test_all_24_ablation_combinations_train_and_reload(self):
        criterion = LossFunction("gaussian", {"name": "baseline"})
        labels = set()
        for c4, c5, encoding, gated in itertools.product(
            ("none", "lsk", "litemla"), ("none", "c2psa"),
            ("binary_slices", "rich8"), (False, True),
        ):
            with self.subTest(c4=c4, c5=c5, encoding=encoding, gated=gated):
                cfg = resolve_ablation_config(
                    self.config, bev_encoding=encoding, scale_gated_fpn=gated,
                    c5_attention=c5,
                )
                cfg["model"]["c4_attention"] = c4
                labels.add(ablation_label(cfg))
                model = build_model(cfg).train()
                channels = 8 if encoding == "rich8" else 35
                sample = (torch.rand(2, channels, 32, 48) > 0.9).float()
                predictions = model(sample)
                self.assertEqual({k: tuple(v.shape) for k, v in predictions.items()}, {
                    "cls": (2, 3, 8, 12), "offset": (2, 2, 8, 12),
                    "size": (2, 2, 8, 12), "yaw": (2, 2, 8, 12),
                })
                targets = {k: torch.zeros_like(v) for k, v in predictions.items()}
                targets["cls"][:, 0, 3, 4] = 1
                targets["yaw"][:, 0] = 1
                targets["reg_mask"] = torch.zeros(2, 8, 12)
                targets["reg_mask"][:, 3, 4] = 1
                loss = criterion(predictions, targets)["loss"]
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                if c4 != "none":
                    gradients = [p.grad for p in model.backbone.c4_attention.parameters()]
                    self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                optimizer.step()
                state = io.BytesIO()
                torch.save(model.state_dict(), state)
                state.seek(0)
                reloaded = build_model(cfg).eval()
                reloaded.load_state_dict(torch.load(state, weights_only=True), strict=True)
                model.eval()
                with torch.no_grad():
                    first, second = model(sample), reloaded(sample)
                for key in first:
                    torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        self.assertEqual(len(labels), 24)

    def test_c4_presets_share_the_rich8_sgfpn_profile(self):
        configs = {
            mode: read_json(CONFIG_DIR / f"kitti_mobilepixor_c4_{mode}.json")
            for mode in ("lsk", "litemla")
        }
        for mode, config in configs.items():
            with self.subTest(mode=mode):
                self.assertEqual(config["data"]["bev_encoding"]["name"], "rich8")
                self.assertIs(config["model"]["scale_gated_fpn"], True)
                self.assertEqual(config["model"]["c4_attention"], mode)
                self.assertEqual(config["model"]["c5_attention"], "none")
                self.assertEqual(config["train"]["physical_batch_size"], 16)
                self.assertEqual(config["train"]["accumulation_steps"], 2)
                self.assertEqual(config["val"]["physical_batch_size"], 16)

        normalized_lsk = deepcopy(configs["lsk"])
        normalized_lsk["model"]["c4_attention"] = "none"
        normalized_lsk["note"] = configs["litemla"]["note"]
        normalized_litemla = deepcopy(configs["litemla"])
        normalized_litemla["model"]["c4_attention"] = "none"
        differences = config_differences(normalized_lsk, normalized_litemla)
        self.assertFalse(
            differences,
            "C4_LSK and C4_LITEMLA must use the same Rich8 + SG-FPN profile; "
            "unexpected fields:\n- " + "\n- ".join(differences),
        )

    def test_all_backbone_branch_configs_share_universal_model_schema(self):
        required_keys = [
            "backbone", "backbone_out_dim", "c4_attention", "c5_attention",
            "cls_encoding", "scale_gated_fpn", "c2psa", "lsk", "litemla",
        ]
        allowed_keys = set(required_keys) | {"c4_attention_route"}
        config_paths = sorted(CONFIG_DIR.glob("*.json"))
        self.assertEqual(len(config_paths), 8)
        for path in config_paths:
            with self.subTest(config=path.name):
                model = read_json(path)["model"]
                self.assertTrue(set(required_keys).issubset(model))
                self.assertTrue(set(model).issubset(allowed_keys))

    def test_config_overrides_do_not_mutate_presets(self):
        original = deepcopy(self.config)
        resolved = resolve_ablation_config(self.config, bev_encoding="rich8", scale_gated_fpn=True)
        self.assertEqual(original, self.config)
        self.assertNotEqual(config_digest(original), config_digest(resolved))
        self.assertEqual(config_digest(original), config_digest(json.loads(json.dumps(original))))

    def test_invalid_switches_fail_loudly(self):
        for mode in ("typo", None, True):
            with self.assertRaises(ValueError):
                build_c4_attention(mode)
        for kwargs in ({"head_dim": 3}, {"head_dim": True}, {"scales": [4]},
                       {"scales": [3, 3]}, {"eps": 0}, {"layer_scale_init": -1}):
            with self.assertRaises(ValueError):
                LiteMLARefinement(**kwargs)
        for kwargs in ({"bev_encoding": "rich11"}, {"scale_gated_fpn": "false"},
                       {"c5_attention": "coordatt"},
                       {"c4_attention_route": "parallel"}):
            with self.assertRaises(ValueError):
                resolve_ablation_config(self.config, **kwargs)
        with self.assertRaises(ValueError):
            MobilePixorBackBone(c4_attention_route="parallel")

    def test_decoupled_and_shared_configs_get_distinct_labels(self):
        shared = deepcopy(self.config)
        shared["model"]["c4_attention"] = "litemla"
        shared["model"]["c5_attention"] = "c2psa"
        decoupled = deepcopy(shared)
        decoupled["model"]["c4_attention_route"] = "lateral_only"
        self.assertNotEqual(ablation_label(shared), ablation_label(decoupled))
        self.assertTrue(ablation_label(decoupled).endswith("_c4route-lateral"))

    def test_decoupled_preset_matches_its_no_attention_control(self):
        control = read_json(
            CONFIG_DIR / "kitti_mobilepixor_rich8_sgfpn_control.json"
        )
        variants = {
            "c4_lateral": read_json(
                CONFIG_DIR / "kitti_mobilepixor_c4_litemla_lateral_only.json"
            ),
            "joint_shared": read_json(
                CONFIG_DIR / "kitti_mobilepixor_c4_litemla_c5_c2psa_shared.json"
            ),
            "joint_lateral": read_json(
                CONFIG_DIR
                / "kitti_mobilepixor_c4_litemla_c5_c2psa_decoupled.json"
            ),
        }
        self.assertEqual(control["model"]["c4_attention"], "none")
        self.assertEqual(control["model"]["c5_attention"], "none")
        expected = {
            "c4_lateral": ("litemla", "none", "lateral_only"),
            "joint_shared": ("litemla", "c2psa", "shared"),
            "joint_lateral": ("litemla", "c2psa", "lateral_only"),
        }
        for name, variant in variants.items():
            with self.subTest(variant=name):
                actual = (
                    variant["model"]["c4_attention"],
                    variant["model"]["c5_attention"],
                    variant["model"]["c4_attention_route"],
                )
                self.assertEqual(actual, expected[name])
                normalized = deepcopy(variant)
                normalized["model"]["c4_attention"] = "none"
                normalized["model"]["c5_attention"] = "none"
                normalized["model"]["c4_attention_route"] = "shared"
                normalized["note"] = control["note"]
                self.assertFalse(config_differences(normalized, control))

    def test_decoupled_preset_backpropagates_through_both_branches(self):
        cfg = read_json(
            CONFIG_DIR / "kitti_mobilepixor_c4_litemla_c5_c2psa_decoupled.json"
        )
        model = build_model(cfg).train()
        outputs = model(torch.randn(2, 8, 32, 48))
        sum(value.square().mean() for value in outputs.values()).backward()
        for module in (model.backbone.c4_attention, model.backbone.c5_attention):
            gradients = [p.grad for p in module.parameters()]
            self.assertTrue(
                all(g is not None and torch.isfinite(g).all() for g in gradients)
            )

    def test_cpu_bfloat16_autocast_backward(self):
        for mode in ("lsk", "litemla"):
            with self.subTest(mode=mode):
                module = build_c4_attention(mode).train()
                sample = torch.randn(2, 64, 9, 11, requires_grad=True)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    output = module(sample)
                    loss = output.float().square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(torch.isfinite(sample.grad).all())
                for parameter in module.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_autocast_backward(self):
        for mode, dtype in itertools.product(("lsk", "litemla"), (torch.float16, torch.bfloat16)):
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            with self.subTest(mode=mode, dtype=dtype):
                module = build_c4_attention(mode).cuda().train()
                sample = torch.randn(2, 64, 12, 14, device="cuda", requires_grad=True)
                with torch.autocast("cuda", dtype=dtype):
                    output = module(sample)
                    loss = output.float().square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(torch.isfinite(sample.grad).all())
                for parameter in module.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    @unittest.skipUnless(
        importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"),
        "Optional ONNX/ONNX Runtime packages are not installed",
    )
    def test_onnx_export_and_runtime_match_all_heads(self):
        import onnx
        import onnxruntime as ort
        from export_onnx import RawHeadWrapper

        for mode, route in itertools.product(
            ("lsk", "litemla"), ("shared", "lateral_only")
        ):
            with self.subTest(mode=mode, route=route):
                cfg = resolve_ablation_config(
                    self.config, bev_encoding="rich8", scale_gated_fpn=True,
                    c5_attention="c2psa", c4_attention_route=route,
                )
                cfg["model"]["c4_attention"] = mode
                model = build_model(cfg).eval()
                # Make the adapter's effect substantial for the parity check.
                with torch.no_grad():
                    model.backbone.c4_attention.layer_scale.fill_(1.0)
                wrapper = RawHeadWrapper(model).eval()
                sample = torch.randn(1, 8, 32, 48)
                stream = io.BytesIO()
                with torch.no_grad():
                    expected = wrapper(sample)
                    torch.onnx.export(
                        wrapper, sample, stream, opset_version=17,
                        input_names=["voxel"], output_names=list(wrapper.OUTPUT_NAMES),
                        do_constant_folding=True, dynamo=False,
                    )
                graph = onnx.load_model_from_string(stream.getvalue())
                onnx.checker.check_model(graph)
                self.assertTrue(any("c4_attention" in node.name for node in graph.graph.node))
                options = ort.SessionOptions()
                options.intra_op_num_threads = 2
                session = ort.InferenceSession(
                    stream.getvalue(), sess_options=options, providers=["CPUExecutionProvider"]
                )
                actual = session.run(None, {"voxel": sample.numpy()})
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(torch.from_numpy(got), want, rtol=2e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
