"""Read-only configured KITTI and train-only GT database auditing."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from benchmark_fixtures import ROOT, asset_fixture


def audit_module():
    assert (ROOT/'tools/benchmarks/audit_kitti_assets.py').is_file(), 'Executable KITTI asset audit is missing'
    return importlib.import_module('tools.benchmarks.audit_kitti_assets')


def test_real_v1_database_and_assets_audit_is_read_only_and_deterministic(tmp_path):
    _, path, processed, raw=asset_fixture(tmp_path)
    before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    module=audit_module()
    result=module.audit_assets(path,kitti_root=raw)
    assert result==module.audit_assets(path,kitti_root=raw.parent)
    assert result['status']=='passed' and result['version']==1
    assert result['metric_mode']=='local_bev'
    assert result['splits']['train']['count']==2 and result['splits']['val']['count']==1
    assert result['databases'][0]['counts']=={'Car':2,'Pedestrian':2,'Cyclist':2}
    assert result['checks']['all_points_finite'] and result['checks']['database_source_labels_match']
    assert len(result['asset_files'])==17 and len(result['inventory_sha256'])==64
    assert {p:p.read_bytes() for p in before}==before
    assert result['processed_root']==str(processed.resolve())


@pytest.mark.parametrize('problem',['point_length','point_nan','label_dimension','label_class','db_source_hash',
    'db_validation_id','db_counts','db_name','db_index','db_geometry','crop_nan','crop_size','bad_calibration',
    'duplicate_ids','overlap','missing_raw_label'])
def test_corrupt_assets_and_database_provenance_are_rejected(tmp_path,problem):
    _,path,processed,raw=asset_fixture(tmp_path)
    info=processed/'gt_database/dbinfos_train.json';db=json.loads(info.read_text())
    if problem=='point_length':(processed/'pointcloud/000000.bin').write_bytes(b'bad')
    elif problem=='point_nan':np.array([np.nan,0,0,1],np.float32).tofile(processed/'pointcloud/000000.bin')
    elif problem=='label_dimension':(processed/'label/000000.txt').write_text('Car -1 1 1 5 0 -1 0\n')
    elif problem=='label_class':(processed/'label/000000.txt').write_text('unknown 1 1 1 5 0 -1 0\n')
    elif problem=='db_source_hash':db['source_manifest_sha256']='0'*64
    elif problem=='db_validation_id':db['db_infos']['Car'][0]['image_idx']='000002'
    elif problem=='db_counts':db['counts']['Car']=999
    elif problem=='db_name':db['db_infos']['Car'][0]['name']='Cyclist'
    elif problem=='db_index':db['db_infos']['Car'][0]['gt_idx']=99
    elif problem=='db_geometry':db['db_infos']['Car'][0]['box3d_lidar'][0]=999.
    elif problem=='crop_nan':np.full((8,4),np.nan,np.float32).tofile(processed/db['db_infos']['Car'][0]['path'])
    elif problem=='crop_size':(processed/db['db_infos']['Car'][0]['path']).write_bytes(b'bad')
    elif problem=='bad_calibration':(raw/'calib/000002.txt').write_text('R0_rect: nan\n')
    elif problem=='duplicate_ids':(tmp_path/'train.txt').write_text('000000;kitti\n000000;kitti\n')
    elif problem=='overlap':(tmp_path/'val.txt').write_text('000001;kitti\n')
    else:(raw/'label_2/000002.txt').unlink()
    if problem.startswith('db_'):info.write_text(json.dumps(db))
    with pytest.raises((ValueError,FileNotFoundError)):
        audit_module().audit_assets(path,kitti_root=raw)


def test_alternate_database_manifest_requires_matching_train_ids_and_source_hash(tmp_path):
    _,path,processed,raw=asset_fixture(tmp_path)
    source=tmp_path/'database_source.txt';source.write_text('000001\n000000\n')
    info=processed/'gt_database/dbinfos_train.json';db=json.loads(info.read_text())
    import hashlib
    db['source_manifest_sha256']=hashlib.sha256(source.read_bytes()).hexdigest();info.write_text(json.dumps(db))
    with pytest.raises(ValueError,match='manifest|source'):
        audit_module().audit_assets(path,kitti_root=raw)
    result=audit_module().audit_assets(path,kitti_root=raw,database_source_split=source)
    assert result['databases'][0]['source_manifest_sha256']==db['source_manifest_sha256']
    source.write_text('000000\n000002\n')
    with pytest.raises(ValueError,match='train|source|manifest'):
        audit_module().audit_assets(path,kitti_root=raw,database_source_split=source)


@pytest.mark.parametrize('mode',['bev','3d'])
def test_reference_audit_requires_projection_calibration_and_image_inputs(tmp_path,mode):
    _,path,_,raw=asset_fixture(tmp_path,box_mode='3d')
    result=audit_module().audit_assets(path,kitti_root=raw,metric_mode=mode)
    assert result['checks']['reference_projection_inputs']
    (raw/'image_2/000002.png').unlink()
    with pytest.raises(FileNotFoundError):audit_module().audit_assets(path,kitti_root=raw,metric_mode=mode)


def test_local_bev_audit_does_not_require_reference_images(tmp_path):
    _,path,_,raw=asset_fixture(tmp_path)
    (raw/'image_2/000002.png').unlink()
    assert audit_module().audit_assets(path,kitti_root=raw)['status']=='passed'


def test_stale_report_is_rejected_by_current_asset_verification(tmp_path):
    _,path,processed,raw=asset_fixture(tmp_path)
    module=audit_module();report=module.audit_assets(path,kitti_root=raw)
    assert module.verify_asset_audit(report,path,metric_mode='local_bev')
    points=processed/'pointcloud/000001.bin'
    values=np.fromfile(points,np.float32);values[-1]=.7;values.tofile(points)
    with pytest.raises(ValueError,match='stale|changed|match'):
        module.verify_asset_audit(report,path,metric_mode='local_bev')
