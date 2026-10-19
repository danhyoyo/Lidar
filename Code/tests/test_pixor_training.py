"""Synthetic geometry, leakage and notebook contracts for PIXOR reproduction."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
SPEC = importlib.util.spec_from_file_location('pixor_training', ROOT / 'tools/pixor_training.py')
pixor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pixor)

SMALL = dict(x_min=0., x_max=8., y_min=-4., y_max=4., z_min=-2.5, z_max=1., resolution=.1, stride=4)


def test_bev_occupancy_reflectance_edges_and_nonfinite_points():
    points = np.array([[1.01, .01, -2.49, .2], [1.01, .01, -2.49, .8],
                       [8., 0, 0, .5], [0, 4, 0, .5], [1, 0, np.nan, .5]], dtype=np.float32)
    bev = pixor.encode_bev(points, SMALL)
    assert bev.shape == (36, 80, 80)
    assert bev[0].sum() == 1
    assert bev[-1, 40, 10] == pytest.approx(.8)
    assert bev[:-1].sum() == 1


def test_rotated_target_and_decoder_round_trip_and_border_ignore():
    box = dict(name='Car', x=4., y=0., length=3., width=1.6, yaw=.6)
    cls, reg, valid = pixor.make_targets([box], SMALL)
    assert cls.sum() > 0
    assert np.any(valid == 0)
    logits = np.where(cls[0] > 0, 12., -12.)
    boxes = pixor.decode_arrays(logits, reg, SMALL, .9, 100, valid[0])
    assert len(boxes) > 0
    np.testing.assert_allclose(boxes[:, :5], np.tile([4., 0., 3., 1.6, .6], (len(boxes), 1)), atol=1e-5)


def test_empty_scene_and_neighbor_class_are_handled_without_false_targets():
    cls, reg, valid = pixor.make_targets([], SMALL)
    assert cls.sum() == 0 and reg.sum() == 0 and valid.min() == 1
    box = dict(name='Van', x=4., y=0., length=3., width=1.6, yaw=0.)
    cls, reg, valid = pixor.make_targets([box], SMALL)
    assert cls.sum() == 0 and np.any(valid == 0)
    assert pixor.decode_arrays(np.full((20, 20), -20.), reg, SMALL, .1, 50).shape == (0, 6)


def test_calibration_box_transform_and_export_uses_kitti_dimension_order(tmp_path):
    # camera (right, down, forward) = (-lidar_y, -lidar_z, lidar_x)
    calib = {'transform': np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1.]]),
             'projection': np.array([[100, 0, 50, 0], [0, 100, 50, 0], [0, 0, 1, 0.]])}
    label = tmp_path / 'label.txt'
    label.write_text('Car 0 0 0 0 0 100 100 1.5 1.6 4.2 2 1.7 20 0.3\n')
    boxes = pixor.read_objects(label, calib)
    assert boxes[0]['x'] == pytest.approx(20)
    assert boxes[0]['y'] == pytest.approx(-2)
    assert boxes[0]['yaw'] == pytest.approx(-np.pi / 2 - .3)
    box = boxes[0]
    line = pixor.kitti_prediction_row([box['x'], box['y'], box['length'], box['width'], box['yaw'], .8],
                                      calib, [100, 100], dict(height=1.5, bottom_z=-1.7))
    fields = line.split()
    assert len(fields) == 16 and fields[0] == 'Car'
    np.testing.assert_allclose(list(map(float, fields[8:11])), [1.5, 1.6, 4.2], atol=1e-4)
    np.testing.assert_allclose(list(map(float, fields[11:14])), [2, 1.7, 20], atol=1e-4)
    assert float(fields[14]) == pytest.approx(.3, abs=1e-4)


def test_reflection_rotation_and_scale_transform_points_and_boxes_consistently():
    points = np.array([[4., 1., 0., .5]], dtype=np.float32)
    box = dict(name='Car', x=4., y=1., length=4., width=2., yaw=.3)
    matrix = np.array([[0., -2., 1.], [-2., 0., -1.], [0., 0., 1.]])
    transformed_points, boxes = pixor.transform_scene(points, [box], matrix)
    np.testing.assert_allclose(transformed_points[0, :2], [-1., -9.])
    assert boxes[0]['x'] == -1 and boxes[0]['y'] == -9
    assert boxes[0]['length'] == 8 and boxes[0]['width'] == 4
    heading = matrix[:2, :2] @ [np.cos(.3), np.sin(.3)]
    assert boxes[0]['yaw'] == pytest.approx(np.arctan2(heading[1], heading[0]))
    assert box['length'] == 4  # Source annotation must remain unchanged.


def test_rotated_nms_removes_duplicates_but_keeps_distant_boxes():
    boxes = np.array([[5, 0, 4, 2, .2, .9], [5, 0, 4, 2, .2, .8], [15, 0, 4, 2, .2, .7]])
    kept = pixor.rotated_nms(boxes, .1, 100)
    np.testing.assert_allclose(kept[:, -1], [.9, .7])


@pytest.mark.parametrize('use_metadata', [False, True])
def test_data_preparation_uses_train_only_priors_and_no_val_targets(tmp_path, monkeypatch, use_metadata):
    import hashlib
    import shutil
    import struct
    import openpcdet_benchmark as common
    work = tmp_path / 'work'
    raw = tmp_path / 'raw/training'
    splits = {'train': ['000000'], 'val': ['000001']}
    (work / 'splits').mkdir(parents=True)
    monkeypatch.setattr(common, 'SPLIT_HASHES', {
        name: hashlib.sha256(('\n'.join(ids) + '\n').encode()).hexdigest() for name, ids in splits.items()})
    for name, ids in splits.items():
        (work / 'splits' / (name + '.txt')).write_text('\n'.join(ids) + '\n')
    (work / 'kitti_reference').mkdir()
    for name in common.EVALUATOR_HASHES:
        shutil.copyfile(ROOT / 'tools/kitti_training_pipeline/kitti_reference' / name, work / 'kitti_reference' / name)
    calibration = ('P2: 100 0 50 0 0 100 50 0 0 0 1 0\nR0_rect: 1 0 0 0 1 0 0 0 1\n'
                   'Tr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\n')
    png = b'\x89PNG\r\n\x1a\n' + b'\0\0\0\rIHDR' + struct.pack('>II', 100, 100)
    for identifier, height in [('000000', 1.5), ('000001', 5.)]:
        for folder, suffix, data in [
            ('velodyne', '.bin', struct.pack('<ffff', 20, -2, -1.7, .5)),
            ('calib', '.txt', calibration.encode()), ('image_2', '.png', png),
            ('label_2', '.txt', f'Car 0 0 0 0 0 100 100 {height} 1.6 4.2 2 1.7 20 0\n'.encode())]:
            if use_metadata and folder == 'image_2':
                continue
            path = raw / folder / (identifier + suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    settings = {**pixor.default_settings(), 'work_dir': str(work), 'output_root': str(tmp_path / 'output'),
                'kitti_root': str(raw)}
    if use_metadata:
        path = raw / 'kitti_image_sizes.json'
        path.write_text(json.dumps(dict(format='kitti_image_sizes_v1', split='training', camera='image_2',
            image_count=2, frames={i: dict(width=100, height=100) for ids in splits.values() for i in ids})))
        settings['image_sizes_json'] = str(path)
    pixor.prepare(settings)
    assets = json.loads((work / 'assets.json').read_text())
    assert assets['projection_prior']['height'] == 1.5
    assert assets['projection_prior']['bottom_z'] == pytest.approx(-1.7)
    assert 'boxes' in assets['frames']['000000']
    assert set(assets['frames']['000001']) == {'image_shape'}
    assert not (Path(assets['training_root']) / 'image_2').exists()
    assert (Path(assets['training_root']) / 'velodyne/000001.bin').exists()
    item = pixor.TrainingDataset(assets, {**settings, 'augmentation': False})[0]
    assert item[0].shape == (36, 800, 700)
    assert item[1].sum() > 0


def test_resume_identity_rejects_training_changes_but_allows_io_changes():
    settings = pixor.default_settings()
    initial = pixor.training_identity(settings, 'dataset-hash')
    assert pixor.training_identity({**settings, 'workers': 0, 'work_dir': '/tmp/new'}, 'dataset-hash') == initial
    for patch in ({'epochs': 51}, {'batch_size': 1}, {'seed': 43}, {'augmentation': False}):
        assert pixor.training_identity({**settings, **patch}, 'dataset-hash') != initial
    assert pixor.training_identity(settings, 'different-data') != initial


def test_notebook_is_standalone_with_canonical_split_and_fifty_epochs(tmp_path, monkeypatch):
    import openpcdet_benchmark as common
    notebook = json.loads((ROOT / 'PIXOR_KITTI_Train_50Epochs.ipynb').read_text())
    monkeypatch.chdir(tmp_path)
    code = next(''.join(c['source']) for c in notebook['cells']
                if c['cell_type'] == 'code' and 'SETTINGS = dict(' in ''.join(c['source']))
    code = code.replace("ON_COLAB = 'COLAB_RELEASE_TAG' in os.environ or Path('/content').is_dir()", 'ON_COLAB = False')
    namespace = {}
    exec(code, namespace)
    assert namespace['SETTINGS']['epochs'] == 50
    embedded = next(''.join(c['source']) for c in notebook['cells']
                    if 'embedded-source' in c.get('metadata', {}).get('tags', []))
    bundle_namespace = {'WORK_DIR': tmp_path}
    exec(embedded, bundle_namespace)
    bundle = bundle_namespace['BUNDLE']
    assert bundle['pixor_training.py'] == (ROOT / 'tools/pixor_training.py').read_text()
    assert bundle['openpcdet_benchmark.py'] == (ROOT / 'tools/openpcdet_benchmark.py').read_text()
    for relative, content in bundle.items():
        assert (tmp_path / relative).read_text() == content
    for name, count in [('train', 3712), ('val', 3769)]:
        file = tmp_path / (name + '.txt')
        file.write_text(bundle['splits/' + name + '.txt'])
        ids = common.read_ids(file)
        assert len(ids) == count
        import hashlib
        assert hashlib.sha256(('\n'.join(ids) + '\n').encode()).hexdigest() == common.SPLIT_HASHES[name]
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<pixor-notebook>', 'exec')
            assert cell.get('outputs', []) == []


def test_model_forward_loss_backward_and_empty_targets_on_cpu():
    torch = pytest.importorskip('torch')
    model = pixor.build_model()
    model.eval()
    # An odd output width exercises the transpose-convolution output_size path.
    inputs = torch.randn(1, 36, 80, 70)
    prediction = model(inputs)
    assert prediction.shape == (1, 7, 20, 18)
    targets = torch.zeros_like(prediction)
    valid = torch.ones(1, 1, 20, 18)
    loss, cls, reg = pixor.compute_loss(prediction, targets[:, :1], targets[:, 1:], valid)
    assert torch.isfinite(loss) and reg.item() == 0
    loss.backward()
    assert model.header[-1].weight.grad is not None
    model.zero_grad(set_to_none=True)
    classification = torch.zeros(1, 1, 20, 18)
    classification[0, 0, 10, 8] = 1
    loss, _, _ = pixor.compute_loss(model(inputs), classification, targets[:, 1:], valid)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.header[-1].weight.grad[1:].abs().sum() > 0
    # Check the actual 800x700 layout without allocating a full training graph.
    with torch.device('meta'):
        full = pixor.build_model().eval()(torch.empty(1, 36, 800, 700))
    assert full.shape == (1, 7, 200, 175)


@pytest.fixture
def cpu_smoke(monkeypatch):
    """Run the real smoke/loss/optimizer with CPU FP16 instead of CUDA I/O."""
    torch = pytest.importorskip('torch')

    class Tiny(torch.nn.Conv2d):
        def cuda(self):
            return self

    model = Tiny(1, 7, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.fill_(-2.)
        model.bias[0] = -4.595
    forwards, scalers = [], []
    model.register_forward_hook(lambda *args: forwards.append(True))
    autocast, grad_scaler = torch.autocast, torch.amp.GradScaler

    def cpu_scaler(device, **kwargs):
        scaler = grad_scaler('cpu', **kwargs)
        scalers.append(scaler)
        return scaler

    monkeypatch.setattr(torch, 'autocast', lambda device, **kwargs: autocast('cpu', **kwargs))
    monkeypatch.setattr(torch.amp, 'GradScaler', cpu_scaler)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, **kwargs: self)
    monkeypatch.setattr(pixor, 'configure_runtime', lambda seed: None)
    monkeypatch.setattr(pixor, 'reference_check', lambda settings: None)
    monkeypatch.setattr(pixor, 'load_assets', lambda settings: {})
    monkeypatch.setattr(pixor, 'build_model', lambda: model)
    batch = (torch.ones(1, 1, 1, 1), torch.ones(1, 1, 1, 1),
             torch.zeros(1, 6, 1, 1), torch.ones(1, 1, 1, 1))
    monkeypatch.setattr(pixor, 'data_loader', lambda *args, **kwargs: [batch])
    return model, pixor.default_settings(), forwards, scalers


def test_smoke_retries_real_fp16_overflow_until_optimizer_updates(cpu_smoke, capsys):
    torch = pytest.importorskip('torch')
    model, settings, forwards, scalers = cpu_smoke
    pixor.smoke(settings)
    # At 65536 the Smooth-L1 derivative overflows FP16; at 32768 it fits.
    assert len(forwards) == 2
    assert scalers[0].get_scale() == 32768
    torch.testing.assert_close(model.bias[1:], torch.full((6,), -1.99))
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert 'Smoke OK' in capsys.readouterr().out


def test_smoke_persistent_nonfinite_gradients_fail_without_success(cpu_smoke, capsys):
    torch = pytest.importorskip('torch')
    model, settings, forwards, scalers = cpu_smoke
    before = model.weight.detach().clone()
    model.weight.register_hook(lambda gradient: torch.full_like(gradient, float('inf')))
    with pytest.raises(FloatingPointError, match='Smoke.*AMP'):
        pixor.smoke(settings)
    assert 1 < len(forwards) <= 32
    assert scalers[0].get_scale() < 65536
    torch.testing.assert_close(model.weight, before)
    assert 'Smoke OK' not in capsys.readouterr().out


@pytest.mark.parametrize('nonfinite', [False, True])
def test_smoke_fp32_keeps_strict_gradient_validation(cpu_smoke, nonfinite):
    torch = pytest.importorskip('torch')
    model, settings, forwards, _ = cpu_smoke
    settings['amp'] = False
    if nonfinite:
        model.weight.register_hook(lambda gradient: torch.full_like(gradient, float('inf')))
        with pytest.raises(RuntimeError, match='non-finite'):
            pixor.smoke(settings)
    else:
        pixor.smoke(settings)
        torch.testing.assert_close(model.bias[1:], torch.full((6,), -1.99))
    assert len(forwards) == 1


def test_resume_completes_interrupted_validation_without_retraining_epoch(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    class Tiny(torch.nn.Conv2d):
        def cuda(self):
            return self
    monkeypatch.setattr(pixor, 'build_model', lambda: Tiny(1, 7, 1))
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, **kwargs: self)
    monkeypatch.setattr(pixor, 'configure_runtime', lambda seed: None)
    monkeypatch.setattr(pixor, 'reference_check', lambda settings: None)
    monkeypatch.setattr(pixor, 'load_assets', lambda settings: {'asset_signature': 'synthetic'})
    trained_epochs = []
    def loader(assets, settings, epoch):
        trained_epochs.append(epoch + 1)
        return [(torch.zeros(2, 1, 1, 1), torch.ones(2, 1, 1, 1),
                 torch.zeros(2, 6, 1, 1), torch.ones(2, 1, 1, 1))]
    monkeypatch.setattr(pixor, 'data_loader', loader)
    evaluations = []
    def validation(model, settings, assets, destination):
        evaluations.append(Path(destination).name)
        if len(evaluations) == 1:
            raise RuntimeError('simulated runtime interruption during validation')
        return {'accuracy': {'R40': {'Moderate': 40. + len(evaluations)}}}
    monkeypatch.setattr(pixor, 'evaluate_model', validation)
    settings = {**pixor.default_settings(), 'work_dir': str(tmp_path / 'work'),
                'output_root': str(tmp_path), 'run_name': 'resume', 'epochs': 2,
                'amp': False, 'accumulation_steps': 1, 'eval_every': 1}
    with pytest.raises(RuntimeError, match='simulated'):
        pixor.train(settings)
    checkpoint = tmp_path / 'resume/checkpoints/last.pth'
    assert torch.load(checkpoint, weights_only=True)['validation_pending'] is True
    pixor.train(settings)
    assert trained_epochs == [1, 2]
    assert evaluations == ['epoch_001', 'epoch_001', 'epoch_002']
    state = torch.load(checkpoint, weights_only=True)
    assert state['epoch'] == 2 and state['validation_pending'] is False
    assert [row['epoch'] for row in state['history']] == [1, 2]
    assert (checkpoint.parent / 'best.pth').exists()
