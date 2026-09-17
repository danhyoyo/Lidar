# MobileBEV-Lite implementation gate

Status: **not evaluated**. The architecture, 3D target/decode path, BEV/3D
local evaluator, checkpoint selector and four ablation configs are implemented.
A4 has 593,075 parameters and all local synthetic checks pass.

No accuracy, latency, bootstrap, ONNX/TensorRT or go/no-go claim is reported:
this checkout has no KITTI data, the available PyTorch environments report no
CUDA device, `onnx` is absent and `trtexec` is unavailable. Run the smoke and
full protocols in `docs/mobile_bev_lightweight/PLAN.md`; replace this status
only with metrics traceable to their config, split and checkpoint hashes.

The novelty position remains deliberately narrow: RichBEV/high-resolution
fusion are prior directions (including TriBand-BEV); the candidate contribution
is the minimal RichBEV-8 + two residual SG-FPN gates + direct `z/h` extension of
the existing IRBGHR-MobilePIXOR pipeline. Do not claim SOTA, first, or
edge-ready from the current evidence.
