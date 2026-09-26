**Modern BEVNeXt Backbone Implementation Plan**  
***For agentic workers:*** * REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (* *- [ ]* *) syntax for tracking.*  
**Goal:** Refactor the outdated MobileNetV2-based backbone in MobilePIXOR into a modern, lightweight 3D LiDAR BEV backbone ("BEVNeXt") featuring 7x7 depthwise convolutions, SiLU activations, single-stage C4 LiteMLA attention, and bilinear scale-gated FPN under 1.0M parameters.  
**Architecture:**  
- Stem: Fast spatial downsampler (stride 2) mapping 8-channel RichBEV to 32 channels.  
- Stages 2–4: ConvNeXt/UIB-style blocks with 7x7 depthwise convolutions (0.7m metric receptive field), inverted expansion (2.5x), and SiLU non-linearity.  
- Attention: Single-stage LiteMLA linear attention at Stage 3 (C4, 100x88) with identity LayerScale, keeping Stage 4 (C5) pure-convolution to prevent attention interference.  
- FPN Neck: Bilinear upsampling + depthwise refinement + zero-initialized scale-gated fusion (eliminating ConvTranspose2d checkerboard artifacts) outputting 16-channel stride-4 features to the detection header.  
**Tech Stack:** PyTorch 2.11.0, CUDA 12.8, NumPy, pytest.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OYQ1AABSAwY8JoIGqr4Z6Eoiggn9mu0twy8wc1RkAAH9xbdVa7V9PAAB47X4A9C4EIsmYmgsAAAAASUVORK5CYII=)  
**Global Constraints**  
- Parameter count must remain strictly **under 2,000,000 parameters** (< 2.0M).  
- Must seamlessly plug into existing CustomModel via cfg["backbone"] = "bevnext".  
- Must output 16-channel features at stride 4 (200 \times 176 for 800 \times 704 input) to feed Header directly without modifying detection heads.  
- Must support both 8-channel (rich8) and 35-channel (binary_slices) inputs.  
- Backward compatibility: preserve all existing backbones (mobilepixor, pixor, rpn).  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSNhwgJmkPYLLpnRgQU2QtIq6DIze3UGAMBf3Gu1VcfHEQAA3rseaHkEMn1wK7sAAAAASUVORK5CYII=)  
**Proposed Changes**  
**Component 1: Modern BEV Block & Attention Modules**  
***[NEW] *** *detector/core/models/backbones/bevnext_blocks.py*  
Contains:  
- BEVNeXtBlock: 7x7 Depthwise Conv -> BatchNorm2d -> 1x1 Pointwise Expand (2.5x) -> SiLU -> 1x1 Pointwise Project -> LayerScale -> Residual Add.  
- DownsampleBlock: 3x3 Conv stride 2 with BatchNorm2d and SiLU.  
- LiteMLARefinement: Ported and generalized for arbitrary channel widths (96 channels) with FP32 linear attention accumulation and LayerScale.  
**Component 2: Complete BEVNeXt Backbone Architecture**  
***[NEW] *** *detector/core/models/backbones/bevnext.py*  
Contains:  
- BEVNeXtBackbone: Complete backbone integrating Stem, Stage 2 (48ch), Stage 3 (96ch + LiteMLA hook), Stage 4 (128ch), and Bilinear Scale-Gated FPN Neck.  
- Parameter count guarantee: ~1.1M to 1.3M parameters.  
**Component 3: Model Factory & Dispatch Integration**  
***[MODIFY] *** *detector/core/models/model.py*  
- Add "bevnext" option into CustomModel.__init__.  
- Pass input_channels, c4_attention, and scale_gated_fpn options to BEVNeXtBackbone.  
**Component 4: Preset Configuration**  
***[NEW] *** *configs/kitti/backbone_branch/kitti_bevnext_litemla.json*  
- Configuration file using rich8, backbone: "bevnext", c4_attention: "litemla", scale_gated_fpn: true.  
**Component 5: Test Suite & Verification**  
***[NEW] *** *tests/test_bevnext_backbone.py*  
- Shape consistency check across all stages.  
- Parameter count check (< 2.0M parameters).  
- Gradient flow check with backward pass.  
- Comparison test verifying that output dimensions match Header requirements (B \times 16 \times 200 \times 176).  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSNhZscZXlheJwqQgQU2QtIq6DIze3UGAMBf3Gu1VcfXEwAAXrseop8EQrmJduIAAAAASUVORK5CYII=)  
**Task Breakdown**  
**Task 1: Implement Core Building Blocks (**bevnext_blocks.py **)**  
**Files:**  
- Create: detector/core/models/backbones/bevnext_blocks.py  
- Test: tests/test_bevnext_backbone.py  
**Interfaces:**  
- Produces: BEVNeXtBlock(channels, expansion=2.5, layer_scale_init=1e-5)  
- Produces: DownsampleBlock(in_channels, out_channels, stride=2)  
- Produces: LiteMLARefinement(channels, head_dim=16, scales=(5,), layer_scale_init=0.01)  
- **Step 1: Write unit tests for core blocks**  
# In tests/test_bevnext_backbone.py  
 import torch  
 import pytest  
 from core.models.backbones.bevnext_blocks import BEVNeXtBlock, DownsampleBlock, LiteMLARefinement  
   
 def test_bevnext_block_shape_and_grad():  
     blk = BEVNeXtBlock(channels=64)  
     x = torch.randn(2, 64, 50, 44, requires_grad=True)  
     out = blk(x)  
     assert out.shape == x.shape  
     out.sum().backward()  
     assert x.grad is not None  
   
 def test_downsample_block():  
     down = DownsampleBlock(in_channels=32, out_channels=64, stride=2)  
     x = torch.randn(2, 32, 100, 88)  
     out = down(x)  
     assert out.shape == (2, 64, 50, 44)  
   
 def test_litemla_block():  
     attn = LiteMLARefinement(channels=96, head_dim=16)  
     x = torch.randn(2, 96, 50, 44, requires_grad=True)  
     out = attn(x)  
     assert out.shape == (2, 96, 50, 44)  
     out.sum().backward()  
     assert x.grad is not None  
   
- **Step 2: Run test to verify it fails initially**  
   
 Run: /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_bevnext_backbone.py -v  
   
 Expected: FAIL (ModuleNotFoundError)  
- **Step 3: Implement ** **BEVNeXtBlock** **, ** **DownsampleBlock** **, and ** **LiteMLARefinement**  
   
 In detector/core/models/backbones/bevnext_blocks.py:  
   
 Implement with 7x7 depthwise conv, BatchNorm2d, 1x1 conv expansion with SiLU, 1x1 conv projection, and LayerScale.  
- **Step 4: Run test to verify it passes**  
   
 Run: /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_bevnext_backbone.py -v  
   
 Expected: PASS  
- **Step 5: Git commit task 1**  
git add detector/core/models/backbones/bevnext_blocks.py tests/test_bevnext_backbone.py  
 git commit -m "feat(backbone): implement BEVNeXt modern building blocks"  
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSNBCUrfD6LYGNDAgAU2QtIq6DIzW7UHAMBfHGt1V+fXEwAAXrseHDAF/orRG+cAAAAASUVORK5CYII=)  
**Task 2: Implement Complete BEVNeXt Backbone (**bevnext.py **)**  
**Files:**  
- Create: detector/core/models/backbones/bevnext.py  
- Test: tests/test_bevnext_backbone.py  
**Interfaces:**  
- Consumes: BEVNeXtBlock, DownsampleBlock, LiteMLARefinement from bevnext_blocks.py  
- Produces: BEVNeXtBackbone(input_channels=8, backbone_out_dim=16, c4_attention='litemla', scale_gated_fpn=True)  
- **Step 1: Write integration test for BEVNeXt Backbone**  
   
 Add tests checking:  
- Forward pass with input (1, 8, 800, 704) producing (1, 16, 200, 176).  
- Total parameter count < 2,000,000.  
- Switchable attention (none vs litemla).  
- **Step 2: Run test to verify failure**  
   
 Run: /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_bevnext_backbone.py -k test_backbone -v  
   
 Expected: FAIL  
- **Step 3: Implement ** **BEVNeXtBackbone**  
- Stem: Conv3x3 (in -> 32, stride 2) + Conv3x3 (32 -> 32)  
- Stage 2: Downsample (32 -> 48) + 2x BEVNeXtBlock (48ch)  
- Stage 3: Downsample (48 -> 96) + 4x BEVNeXtBlock (96ch) + Optional LiteMLA  
- Stage 4: Downsample (96 -> 128) + 2x BEVNeXtBlock (128ch, no attention)  
- Bilinear FPN Neck: Bilinear upsample 2x + 3x3 depthwise conv + scale gates  
- Output lateral fusion to 16 channels.  
- **Step 4: Run test to verify it passes**  
   
 Run: /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_bevnext_backbone.py -v  
   
 Expected: PASS  
- **Step 5: Git commit task 2**  
git add detector/core/models/backbones/bevnext.py tests/test_bevnext_backbone.py  
 git commit -m "feat(backbone): implement BEVNeXt full backbone with bilinear SG-FPN"  
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSNBCUrfEJoYGDDBgAU2QtIq6DIzW7UHAMBfHGt1V+fXEwAAXrseJfoF+ljayTIAAAAASUVORK5CYII=)  
**Task 3: Integrate into **CustomModel ** and Model Factory**  
**Files:**  
- Modify: detector/core/models/model.py  
- Test: tests/test_bevnext_backbone.py  
**Interfaces:**  
- Consumes: BEVNeXtBackbone from core.models.backbones.bevnext  
- Produces: CustomModel(cfg) when cfg["backbone"] == "bevnext"  
- **Step 1: Write test for CustomModel with ** **backbone: bevnext**  
def test_custom_model_bevnext():  
     cfg = {  
         "backbone": "bevnext",  
         "cls_encoding": "gaussian",  
         "backbone_out_dim": 16,  
         "c4_attention": "litemla",  
         "scale_gated_fpn": True  
     }  
     model = CustomModel(cfg, num_classes=4, input_channels=8)  
     x = torch.randn(1, 8, 800, 704)  
     pred = model(x)  
     assert "cls" in pred and "offset" in pred and "size" in pred and "yaw" in pred  
     assert pred["cls"].shape == (1, 4, 200, 176)  
   
- **Step 2: Run test to verify failure**  
   
 Run: /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_bevnext_backbone.py -k test_custom_model -v  
   
 Expected: FAIL (Unsupported backbone)  
- **Step 3: Modify ** **detector/core/models/model.py**  
   
 Add import and dispatch branch:  
elif cfg["backbone"] == "bevnext":  
     self.backbone = BEVNeXtBackbone(  
         input_channels=input_channels,  
         backbone_out_dim=cfg.get("backbone_out_dim", 16),  
         c4_attention=cfg.get("c4_attention", "litemla"),  
         scale_gated_fpn=cfg.get("scale_gated_fpn", True),  
     )  
   
- **Step 4: Run test to verify it passes**  
   
 Run: /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_bevnext_backbone.py -v  
   
 Expected: PASS  
- **Step 5: Git commit task 3**  
git add detector/core/models/model.py tests/test_bevnext_backbone.py  
 git commit -m "feat(model): register bevnext backbone into CustomModel"  
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OQQ2AQBAAsSE5CbzRujLwhwQMYIEfIWkVdJuZozoDAOAvrlWtav96AgDAa/cDEXQEKquakOYAAAAASUVORK5CYII=)  
**Task 4: Create Preset Config & End-to-End Latency Benchmark**  
**Files:**  
- Create: configs/kitti/backbone_branch/kitti_bevnext_litemla.json  
- Test: Benchmark script measuring latency and VRAM on GPU  
- **Step 1: Create config file**  
   
 Create configs/kitti/backbone_branch/kitti_bevnext_litemla.json using Rich8 BEV encoding, BEVNeXt backbone, LiteMLA attention, and baseline loss settings.  
- **Step 2: Run benchmark script comparing MobilePIXOR vs BEVNeXt**  
   
 Verify:  
- Parameters < 2.0M.  
- Model latency on CUDA.  
- Peak VRAM allocated.  
- **Step 3: Git commit task 4**  
git add configs/kitti/backbone_branch/kitti_bevnext_litemla.json  
 git commit -m "feat(config): add kitti_bevnext_litemla preset configuration"  
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OYQ1AABSAwc8mi5wvkwZyCKCAACr4Z7a7BLfMzFYdAQDwF+da3dX+9QQAgNeuB6feBdUJcyS2AAAAAElFTkSuQmCC)  
**Verification Plan**  
**Automated Tests**  
Run full test suite:  
/home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_bevnext_backbone.py -v  
   
Run existing regression tests to ensure no breaking changes:  
/home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_regressions.py -v  
   
**Manual Verification**  
Execute end-to-end forward/backward benchmark on CUDA:  
/home/duyennh/miniconda3/envs/AI_env/bin/python -c '  
 import torch, sys; sys.path.append("detector")  
 from core.models.model import CustomModel  
 cfg = {"backbone": "bevnext", "cls_encoding": "gaussian", "backbone_out_dim": 16, "c4_attention": "litemla", "scale_gated_fpn": True}  
 m = CustomModel(cfg, num_classes=4, input_channels=8).cuda()  
 p = sum(x.numel() for x in m.parameters())  
 print(f"Total Model Parameters: {p:,} (under 2M: {p < 2000000})")  
 x = torch.zeros(1, 8, 800, 704, device="cuda")  
 out = m(x)  
 print({k: v.shape for k, v in out.items()})  
 '  
   
Confirm:  
1. Total parameters strictly < 2,000,000 (target ~1.2M).  
2. Shapes are all (1, C, 200, 176).  
3. Loss computation and backward gradient pass succeed without NaN or OOM.  
