#!/usr/bin/env bash
set -euo pipefail
ABLATION_PLAN_ROOT="artifacts/ablation_analysis_2026-10-11/reuse_fake_gt"
ABLATION_OUTPUT_ROOT="${1:-artifacts/paper_bev_continuation}"
ABLATION_RAW_KITTI_ROOT="${2:-data/kitti/raw}"
if [[ ! -f tools/kitti_training_pipeline/train.py ]]; then
    echo "Run this script from the repository root." >&2
    exit 1
fi
for ABLATION_VARIANT in P01_hybrid_f24_h32 P02_standard_f24_h32 P03_none_f24_h32; do
    ABLATION_RUN_NAME="paper_${ABLATION_VARIANT}_s42"
    if [[ -e "$ABLATION_OUTPUT_ROOT/$ABLATION_RUN_NAME/run.json" ]]; then
        echo "Existing run: $ABLATION_RUN_NAME. Resume that same run explicitly or run the remaining variants individually." >&2
        exit 1
    fi
    python3 tools/kitti_training_pipeline/train.py \
        --config "$ABLATION_PLAN_ROOT/configs/$ABLATION_VARIANT.json" \
        --detector-root detector --output-root "$ABLATION_OUTPUT_ROOT" \
        --run-name "$ABLATION_RUN_NAME" --seed 42
    python3 tools/kitti_training_pipeline/evaluate_kitti_bev.py \
        --name "$ABLATION_RUN_NAME" --backend pytorch \
        --model "$ABLATION_OUTPUT_ROOT/$ABLATION_RUN_NAME/selected/best_ap.pt" \
        --config "$ABLATION_OUTPUT_ROOT/$ABLATION_RUN_NAME/config.resolved.json" \
        --detector-root detector --kitti-root "$ABLATION_RAW_KITTI_ROOT" \
        --split splits/kitti/val.txt --nms-alpha 0 \
        --output "$ABLATION_OUTPUT_ROOT/$ABLATION_RUN_NAME/evaluation_ap_local_bev.json"
done
