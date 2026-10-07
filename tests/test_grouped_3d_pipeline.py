"""Complete 3D recipes and actual synthetic CPU/CUDA training/resume gates."""

import copy
import importlib
import json
import sys
from pathlib import Path

import pytest
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "detector"), str(ROOT / "detector/core/datasets"),
               str(ROOT / "tools/kitti_training_pipeline")]

from common import checkpoint_identity, detection_spec, generate_run_name, read_json
import notebook_config
from tools.benchmarks.profile_detector import profile_detector
from test_vertical_head import config_3d

MANIFEST = read_json(ROOT / "docs/plans/lightweight_lidar_backbone_2026/experiment_manifest.json")
VARIANTS = [v for v in MANIFEST["variants"] if v["overrides"].get("data",{}).get("box_mode") == "3d"]


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v:v["id"])
def test_complete_3d_recipe_matches_manifest_budget_hybrid_schedule_and_vertical_contract(variant):
    path = ROOT / variant["planned_config"]
    assert path.is_file(), "Complete 3D runtime recipe is missing"
    config = read_json(path)
    original = copy.deepcopy(config)
    assert hasattr(notebook_config,"resolve_under1m_3d_recipe"), "Explicit 3D recipe resolver is missing"
    assert config == notebook_config.resolve_under1m_3d_recipe(ROOT,variant["id"])
    spec = detection_spec(config)
    assert spec.box_mode == "3d" and spec.vertical_loss_weight == 1.
    assert spec.head_mode == "grouped" and spec.use_iou
    base = read_json(ROOT/MANIFEST["base_config"])
    assert all(config[key] == base[key] for key in ("seed","train","val"))
    assert config["augmentation"] == read_json(ROOT/MANIFEST["augmentation_config"])["augmentation"]
    assert config["model"]["local_attention"] == ("none" if variant["id"] == "main_focal_3d" else "eca")
    assert config["model"]["backbone_out_dim"] == 32 and config["model"]["c4_attention"] == "none"
    report = profile_detector(config)
    assert config == original
    assert report["box_mode"] == report["criterion_box_mode"] == "3d"
    counts = report["parameter_counts"]
    assert counts["backbone_including_neck"] == variant["backbone_including_neck_projected"] < 1_000_000
    assert counts["backbone_body"] + counts["neck"] == counts["backbone_including_neck"]
    assert counts["heads"] == 223413 and counts["criterion_train_only"] == 12
    assert counts["total_detector"] == variant["detector_iqa_on_projected"]
    assert report["output_shapes"]["groups"]["ped_cyc"]["vertical"] == [1,2,200,176]
    bev = copy.deepcopy(config);bev["data"]["box_mode"] = "bev"
    assert generate_run_name(config) != generate_run_name(bev)
    assert checkpoint_identity(config) != checkpoint_identity(bev)


def test_explicit_3d_generator_is_deterministic_and_preserves_all_bev_presets(tmp_path):
    assert hasattr(notebook_config,"write_under1m_3d_presets"), "Explicit 3D recipe generator is missing"
    for relative in [MANIFEST["base_config"],MANIFEST["augmentation_config"],notebook_config.UNDER1M_MANIFEST]:
        target = tmp_path/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/relative).read_bytes())
    bev = notebook_config.write_under1m_presets(tmp_path)
    saved = {p:p.read_bytes() for p in bev}
    first = notebook_config.write_under1m_3d_presets(tmp_path)
    snapshots = {p:p.read_bytes() for p in first}
    assert len(first) == 2 and first == notebook_config.write_under1m_3d_presets(tmp_path)
    assert snapshots == {p:p.read_bytes() for p in first}
    assert saved == {p:p.read_bytes() for p in bev}
    assert set(notebook_config.UNDER1M_PRESETS.values()).isdisjoint({v["id"] for v in VARIANTS})


def test_profiler_constructs_the_vertical_criterion_instead_of_a_bev_surrogate():
    report = profile_detector(config_3d(),shape=(1,14,32,48))
    assert report["box_mode"] == report["criterion_box_mode"] == "3d"


@pytest.mark.parametrize("variant",VARIANTS,ids=lambda v:v["id"])
def test_colab_existing_file_override_selects_3d_and_preserves_runtime_data_paths(tmp_path,variant):
    config=notebook_config.resolve_notebook_config(ROOT,preset="config",augmentation="config",
        file_override=variant["planned_config"],runtime_overrides={"data":{"kitti":{"location":str(tmp_path/"processed")}},
            "train":{"precision":"bf16","num_workers":6}})
    assert detection_spec(config).box_mode=="3d" and config["loss"]["vertical_loss_weight"]==1.
    assert config["data"]["kitti"]["location"]==str(tmp_path/"processed")
    assert config["model"]["backbone_out_dim"]==32
    assert config["augmentation"]==read_json(ROOT/MANIFEST["augmentation_config"])["augmentation"]


@pytest.mark.parametrize("local", ["none","eca"])
@pytest.mark.parametrize("backend", ["python","numba"])
def test_actual_hybrid_paste_and_global_scale_supervise_scaled_bottom_z_and_height(tmp_path,local,backend):
    from test_hybrid_augmentation import database, recipe
    from test_grouped_checkpoint import components, payload, restore
    from core.datasets.dataset import Dataset
    from torch.utils.data import DataLoader
    from postprocess import filter_pred_3d
    import train
    train.seed_everything(42)
    root,source,_ = database(tmp_path)
    np.empty((0,4),np.float32).tofile(root/"pointcloud/000000.bin")
    (root/"label/000000.txt").write_text("")
    manifest=tmp_path/"host.txt";manifest.write_text("000000;kitti\n000001;kitti\n")
    config=config_3d();config["model"]["local_attention"]=local
    config["data"]["kitti"].update(location=str(root),geometry={"x_min":0.,"x_max":32.,"x_res":.5,
        "y_min":-8.,"y_max":8.,"y_res":.5,"z_min":-2.5,"z_max":1.,"z_res":.1})
    config["data"]["bev_encoding"]["backend"]="numpy" if backend=="python" else "numba"
    config["augmentation"]={"mode":"openpcdet","AUG_CONFIG_LIST":[recipe(PROBABILITY=1.),
        {"NAME":"random_world_scaling","WORLD_SCALE_RANGE":[1.25,1.25]}]}
    config["train"].update(data=str(manifest),target_backend=backend,precision="fp32",num_workers=0,
                          physical_batch_size=1,accumulation_steps=1,epochs=4,warmup_epochs=0)
    dataset=Dataset(str(manifest),config["data"],config["augmentation"],"gaussian","train",backend)
    batch=next(iter(DataLoader(dataset,batch_size=1,num_workers=0)))
    target=batch["groups"]["car"];mask=target["reg_mask"][0].bool()
    assert mask.any() and batch["voxel"][:,11].sum()>0
    torch.testing.assert_close(target["vertical"][0,0][mask],torch.full_like(target["vertical"][0,0][mask],float(source[6]*1.25)))
    torch.testing.assert_close(target["vertical"][0,1][mask],torch.full_like(target["vertical"][0,1][mask],float(np.log(source[1]*1.25))))
    assert not batch["groups"]["ped_cyc"]["reg_mask"].any()
    parts=components(config);losses=parts[1](parts[0](batch["voxel"]),batch)
    losses["loss"].backward();parts[2].step();parts[3].step()
    checkpoint=tmp_path/"assembled3d.pt";torch.save(payload(config,parts),checkpoint)
    destination=components(config)
    assert restore(torch.load(checkpoint,weights_only=True),config,destination)["mode"]=="resume"
    destination[0].eval()
    with torch.no_grad():prediction=destination[0](batch["voxel"])
    rows=filter_pred_3d(prediction,config["data"]["kitti"],4,.05,.1,task_groups=dataset.task_groups,use_iou=True,max_detections=7)
    assert rows.shape[1]==9 and np.isfinite(rows).all()


@pytest.mark.parametrize("strategy", ["baseline","oga"])
@pytest.mark.parametrize("classification", ["gaussian","binary"])
@pytest.mark.parametrize("iqa", [False,True])
def test_cpu_3d_pipeline_preserves_owned_vertical_training_resume_and_decode(strategy,classification,iqa):
    smoke = importlib.import_module("tools.benchmarks.smoke_detector")
    config = config_3d(strategy=strategy,classification=classification,iqa=iqa)
    original = copy.deepcopy(config)
    report = smoke.run_smoke(config,device="cpu",precisions=["fp32"],steps=2,num_workers=0)
    assert config == original
    run = report["precisions"]["fp32"]
    assert report["box_mode"] == "3d"
    assert run["optimizer_updates"] == 2 and run["finite_gradients"]
    assert run["detection_shape"][1] == 9
    assert len(run["saved_predictions_3d"]) == run["detection_shape"][0]
    assert all(len(row) == 9 for row in run["saved_predictions_3d"])
    assert run["vertical_loss_dtype"] == "torch.float32"
    assert all(run["vertical_head_weight_changed"].values())
    assert run["vertical_gradient_verified"] and run["vertical_target_ownership_verified"]
    assert run["optimizer_state_restore_verified"] and run["scheduler_scaler_rng_restore_verified"]
    assert run["checkpoint_restore_verified"] and run["criterion_state_unchanged_in_validation"]
    assert run["negative_only_group_seen"] and run["empty_scene_seen"]


@pytest.mark.skipif(not torch.cuda.is_available(),reason="Actual CUDA required; skips do not establish readiness")
@pytest.mark.parametrize("local", ["none","eca"])
@pytest.mark.parametrize("strategy,classification,iqa", [
    ("baseline","gaussian",True),("oga","gaussian",True),
    ("baseline","binary",True),("oga","binary",True),
    ("baseline","gaussian",False),("oga","gaussian",False),
    ("baseline","binary",False),("oga","binary",False)])
def test_cuda_3d_advertised_objectives_have_vertical_gradients_and_strict_resume(local,strategy,classification,iqa):
    smoke = importlib.import_module("tools.benchmarks.smoke_detector")
    config = config_3d(strategy=strategy,classification=classification,iqa=iqa)
    config["model"]["local_attention"] = local
    report = smoke.run_smoke(config,device="cuda",precisions=["fp32"],steps=2,num_workers=0)
    run = report["precisions"]["fp32"]
    assert report["status"] == "passed" and run["vertical_gradient_verified"]
    assert all(run["vertical_head_weight_changed"].values())
    assert run["optimizer_state_restore_verified"] and run["scheduler_scaler_rng_restore_verified"]


@pytest.mark.skipif(not torch.cuda.is_available(),reason="Actual CUDA required; skips do not establish readiness")
@pytest.mark.parametrize("local", ["none","eca"])
def test_cuda_3d_main_amp_and_full_resolution_outputs(local):
    if not torch.cuda.is_bf16_supported():pytest.skip("Actual BF16 hardware required")
    smoke = importlib.import_module("tools.benchmarks.smoke_detector")
    config = config_3d();config["model"]["local_attention"] = local
    report = smoke.run_smoke(config,device="cuda",precisions=["fp32","fp16","bf16"],steps=2,num_workers=0,full_resolution=True)
    assert report["status"] == "passed"
    for precision,run in report["precisions"].items():
        assert run["vertical_loss_dtype"] == "torch.float32" and run["vertical_gradient_verified"]
        assert run["full_resolution_output_shapes"]["groups"]["car"]["vertical"] == [1,2,200,176]
        assert run["full_resolution_finite"] and all(run["vertical_head_weight_changed"].values())
