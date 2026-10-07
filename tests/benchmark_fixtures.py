"""Small actual KITTI-format assets for audit/evidence integration tests."""
import copy
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools/kitti_training_pipeline'), str(ROOT / 'detector'), str(ROOT / 'detector/core/datasets')]


def asset_fixture(directory, *, box_mode='bev'):
    from build_gt_database import build_database
    from tools.kitti_training_pipeline.kitti_box_conversion import read_calibration, lidar_box_to_camera
    directory = Path(directory)
    processed = directory / 'processed'
    raw = directory / 'raw/training'
    for p in [processed/'pointcloud', processed/'label', raw/'label_2', raw/'calib', raw/'image_2']:
        p.mkdir(parents=True, exist_ok=True)
    classes = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
    rng = np.random.default_rng(42)
    def chunk(kind, body):
        return struct.pack('>I', len(body)) + kind + body + struct.pack('>I', zlib.crc32(kind+body))
    png = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 128, 128, 8, 2, 0, 0, 0))
           + chunk(b'IDAT', zlib.compress((b'\x00'+b'\x00'*384)*128)) + chunk(b'IEND', b''))
    for index in range(3):
        identifier = f'{index:06d}'
        calib = raw/'calib'/f'{identifier}.txt'
        calib.write_text('P2: 70 0 64 0 0 70 64 0 0 0 1 0\nR0_rect: 1 0 0 0 1 0 0 0 1\n'
                         'Tr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\n')
        calibration = read_calibration(calib)
        lines, original, clouds = [], [], []
        for offset, name in enumerate(classes):
            h, w, length, x, y, z, yaw = 1.7, 1.6, 3.5, 5.+offset*5, 0., -1.7, 0.
            lines.append(f'{name} {h} {w} {length} {x} {y} {z} {yaw}')
            camera = lidar_box_to_camera([x,y,z,length,w,h,yaw], calibration)
            original.append(f'{name} 0 0 0 0 0 100 100 {h} {w} {length} ' + ' '.join(map(str,camera[:3])) + f' {camera[6]}')
            xyz = rng.uniform([x-.1,y-.1,z+.2], [x+.1,y+.1,z+1.], (8,3))
            clouds.append(np.column_stack([xyz,np.full(8,.5)]))
        np.concatenate(clouds).astype(np.float32).tofile(processed/'pointcloud'/f'{identifier}.bin')
        (processed/'label'/f'{identifier}.txt').write_text('\n'.join(lines)+'\n')
        (raw/'label_2'/f'{identifier}.txt').write_text('\n'.join(original)+'\n')
        (raw/'image_2'/f'{identifier}.png').write_bytes(png)
    train, val = directory/'train.txt', directory/'val.txt'
    train.write_text('000000;kitti\n000001;kitti\n'); val.write_text('000002;kitti\n')
    build_database(processed, train, processed/'gt_database', val_manifest=val)
    source = ROOT/'configs/experiments/under1m/hist14_local3_focal_grouped_oga_iqa.json'
    config = copy.deepcopy(json.loads(source.read_text()))
    config['data']['box_mode'] = box_mode
    config['data']['kitti'].update(location=str(processed), geometry={'x_min':0.,'x_max':24.,'x_res':.5,
        'y_min':-16.,'y_max':16.,'y_res':.5,'z_min':-2.5,'z_max':1.,'z_res':.1})
    config['data']['bev_encoding']['backend'] = 'numpy'
    config['train'].update(data=str(train), epochs=2, warmup_epochs=0, precision='fp32',
        target_backend='python', num_workers=0, physical_batch_size=1)
    config['val'].update(data=str(val), physical_batch_size=1)
    if box_mode=='3d':config['loss']['vertical_loss_weight']=1.
    path=directory/'config.json';path.write_text(json.dumps(config))
    return config, path, processed, raw


def evaluation_fixture(directory, *, metric_mode='local_bev', iqa=True):
    """Train two tiny CPU epochs and run the actual selected-checkpoint evaluator."""
    import torch
    from torch.utils.data import DataLoader
    import train
    from common import build_model
    from core.datasets.dataset import Dataset
    from tools.benchmarks.benchmark_protocol import freeze_protocol
    from tools.kitti_training_pipeline.evaluate_kitti_bev import run_evaluation as local_evaluation
    from tools.kitti_training_pipeline.evaluate_kitti_3d import run_evaluation as reference_evaluation
    train.seed_everything(42)
    config,path,processed,raw=asset_fixture(directory,box_mode='bev' if metric_mode=='local_bev' else '3d')
    config['model']['header_use_iou']=iqa
    config['loss']['use_iou']=iqa
    config['model']['c4_context']='none'
    for key in list(config['model']):
        if key.startswith('c4_context_'):del config['model'][key]
    path.write_text(json.dumps(config))
    model=build_model(config);criterion=train.build_training_criterion(config,torch.device('cpu'))
    optimizer=train.build_optimizer(model,criterion,config);scheduler=train.build_scheduler(optimizer,config,2)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    training=Dataset(config['train']['data'],config['data'],config['augmentation'],'gaussian','train','python')
    validation=Dataset(config['val']['data'],config['data'],config['augmentation'],'gaussian','validation','python')
    train_loader=DataLoader(training,batch_size=1,num_workers=0)
    val_loader=DataLoader(validation,batch_size=1,num_workers=0)
    directory=Path(directory);(directory/'best').mkdir();(directory/'selected').mkdir()
    history=[];best=float('inf');best_path=None
    for epoch in (1,2):
        model.train();criterion.train();train.set_loss_epoch(criterion,epoch-1)
        updates=0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True);loss=criterion(model(batch['voxel']),batch)['loss']
            loss.backward();optimizer.step();updates+=1
        scheduler.step()
        values=train.validate(model,criterion,val_loader,torch.device('cpu'),'fp32')
        retained=values['loss']<best
        if retained:best=values['loss']
        history.append({'epoch':epoch,'validation':values,'optimizer_updates':updates,'retained':retained,
                        'precision':'fp32','loss':'oga'})
        destination=directory/'best'/f'{epoch}epoch.pt'
        torch.save(train.checkpoint_payload(model,criterion,optimizer,scheduler,scaler,epoch,values,best,config),destination)
        if retained:best_path=destination
    history_path=directory/'metrics.jsonl';history_path.write_text(''.join(json.dumps(row)+'\n' for row in history))
    selected=directory/'selected/best.pt';selected.write_bytes(best_path.read_bytes())
    best_row=min(history,key=lambda row:row['validation']['loss'])
    selection=directory/'selected/selection.json';selection.write_text(json.dumps({'epoch':best_row['epoch'],
        'validation_objective':best,'criterion':'minimum mean validation loss','checkpoint':str(best_path)}))
    evaluator=local_evaluation if metric_mode=='local_bev' else reference_evaluation
    options={} if metric_mode=='local_bev' else {'metric_mode':metric_mode}
    report_path=directory/'evaluation.json'
    evaluator(name='internal_no_context',backend='pytorch',model_path=selected,config_path=path,detector_root=ROOT/'detector',
        kitti_root=raw if metric_mode!='local_bev' else raw.parent,split_path=Path(config['val']['data']),
        output_path=report_path,device='cpu',warmup_frames=0,progress_every=0,**options)
    protocol=directory/'protocol.json';protocol.write_text(json.dumps(freeze_protocol(path,metric_mode=metric_mode)))
    return protocol,report_path,selection,history_path,path,processed,raw
