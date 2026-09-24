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

    def test_refined_c4_feeds_both_c5_and_lateral(self):
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

    def test_new_presets_are_single_changes(self):
        for mode in ("lsk", "litemla"):
            config = read_json(CONFIG_DIR / f"kitti_mobilepixor_c4_{mode}.json")
            self.assertEqual(config["model"]["c4_attention"], mode)
            config["model"]["c4_attention"] = "none"
            config["model"].pop(mode)
            config["note"] = self.config["note"]
            self.assertEqual(config, self.config)

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
                       {"c5_attention": "coordatt"}):
            with self.assertRaises(ValueError):
                resolve_ablation_config(self.config, **kwargs)

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

        for mode in ("lsk", "litemla"):
            with self.subTest(mode=mode):
                cfg = resolve_ablation_config(
                    self.config, bev_encoding="rich8", scale_gated_fpn=True,
                    c5_attention="c2psa",
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
