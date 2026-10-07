# Milestone B: complete 3D synthetic verification

Task58, branch `research/mobilepixornext-under1m`. This records a functional 3D candidate and its synthetic verification. Final benchmark freeze remains task51; real KITTI data/checkpoints are on Colab and were not inspected remotely. No trained KITTI AP, measured latency, deployment or SOTA result is claimed.

## Final GPU recipes

Two complete resolved configs are materialized from the manifest. Main: hist14 v1, depths3/4/2, focal context, local attention none, SG-FPN output32/fusion24, independent Car/Pedestrian-Cyclist heads, Gaussian OGA+BEV IQA, fixed vertical coefficient1 and existing hybrid GT. ECA is a separate optional comparison. Seed42, schedule50 epochs/4 warmup, AdamW, BF16, batch16, workers6 and split paths are inherited without changing the BEV files.

| Recipe | Body | Neck | Backbone + neck | Heads | Detector | Criterion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Main focal 3D | 629,984 | 30,544 | 660,528 | 223,413 | 883,941 | 12 |
| Optional focal ECA 3D | 629,988 | 30,544 | 660,532 | 223,413 | 883,945 | 12 |

Each head adds its independent `[z_bottom,log(height)]` branch. The explicit 3D resolver/generator leave existing BEV notebook presets and the nine-file BEV generator unchanged. The profiler now constructs the actual 3D criterion and reports model/criterion box modes. The [Colab runbook](colab_3d_runbook.md) uses the existing file override and runtime paths without changing notebook defaults.

## Numerical and state gates

- Actual CPU/CUDA FP32 pipelines cover baseline/OGA × Gaussian/binary × IQA on/off; CUDA covers main and ECA. Other 3D strategies remain rejected. Vertical supervised gradients and head weight changes are verified independently of AdamW weight decay.
- Analytic preview matches every supervised synthetic vertical cell to its label's BEV geometry. Actual hybrid database paste followed by fixed global scale verifies scaled bottom-z/log-height through hist14, optimizer, strict checkpoint restore and Nx9 decoding, for Python/Numba backends and both local options.
- Strict restore checks model/criterion, optimizer moments/groups, scheduler, scaler and Python/NumPy/Torch CPU/CUDA/loader-generator RNG. Validation preserves adaptive state and reloaded evaluation predictions match exactly. Persistent-worker augmentation replay is not promised; smoke augmentation is disabled explicitly.
- Dedicated GPU pytest gate: **30 passed in17.92s** before the four assembled-hybrid and two Colab-resolution cases were added. Actual CLI runs cover main/ECA × workers0/6, with three successful updates in each FP32/FP16/BF16 run: **12 precision runs,36 successful updates**. Every run passes full `[1,14,800,704]` inference, with per-group vertical `[1,2,200,176]`.
- BF16/FP32 skip no optimizer updates. FP16 skips16 initially for main and17 for ECA, ending at scales1 and0.5 respectively. The first short gate incorrectly allowed negative-only successful updates to finish before any finite vertical gradient. Keeping the same supervised synthetic batch across overflow exposed the need for scale below1 in ECA; the bounded 3D attempt budget was increased to32 additional attempts. All skips and bad-gradient parameter names remain recorded. Production trainer/GradScaler defaults/objective formulas are unchanged.
- Peak allocated GPU memory across CLI runs: **111,397,888–140,730,880 bytes**, including tiny synthetic training and full-resolution inference. This is not full-resolution backward, latency or deployment profiling. Rotated NMS uses CPU polygon fallback. No Inductor, Python3.10 or CUDA rotated-NMS claim.

Hardware/software: RTX4050 Laptop, driver610.57.04; Python3.14.4, Torch2.11.0/Torch CUDA13.0, NumPy2.4.6, Numba0.67.0. Reference/full gates use isolated numba-cuda0.30.4 with CUDA12.9 compiler/runtime libraries; training packages are unchanged. Colab must verify its own environment and precision support.

## Reference and protocol boundary

The OpenPCDet revision/hash/license contract from task57 remains unchanged. Saved GPU smoke predictions and their actual synthetic source boxes are converted with explicit synthetic camera matrices and eight-corner projection. Across main/ECA and all three precisions, both AP3D/APBEV and both R11/R40 agree with direct pinned reference entry points. The41 replicated synthetic frames support recall sampling checks; their AP is not trained KITTI accuracy.

The updated primary 3D and reference BEV protocol records use the complete 3D main config, pinned revision, actual smoke/config hashes and reference verification. Both retain `long_training_allowed=false`. Only the **configured real-data/train-only GT database audit** and **comparator protocols/reproduced baseline** remain unresolved. Data locations absent locally do not describe the user's Colab assets. The [Colab runbook](colab_3d_runbook.md) supplies audit/selection/evaluation steps; comparator details still need to be recorded.

Final full/focused suite counts, source preservation and exact artifact links are recorded in [execution progress](execution_progress.md). Ignored GPU reports and saved-reference evidence are under `artifacts/kitti/backbone_audit/batch20/`. The implementation is available for review; complete benchmark readiness requires the remaining external evidence.
