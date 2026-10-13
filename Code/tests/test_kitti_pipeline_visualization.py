"""Actual grouping and forward capture for the end-to-end KITTI exporter."""
import numpy as np
import pytest
import torch

from tools.visualization import export_kitti_pipeline as pipeline


def test_occupied_cells_reconstruct_hist14_height_counts_and_boundaries():
    from core.bev_encoding import resolve_bev_encoding
    from core.datasets.utils_1.bev_backend import encode_hist14_chw
    geometry = dict(x_min=0., x_max=4., x_res=1., y_min=-2., y_max=2., y_res=1.,
                    z_min=-2., z_max=2., z_res=.5)
    schema = resolve_bev_encoding({'name': 'hist14', 'version': 1, 'backend': 'numpy'}, geometry)
    points = np.array([[.2, -.8, -1.5, .5], [.3, -.7, -1.4, .6], [.4, -.7, .5, .2],
                       [1.2, .8, 1.5, .9], [0., .5, 0., .1], [4., .5, 0., .1],
                       [1., 2., 0., .1], [1., 0., -2., .1], [1., 0., np.nan, .1]], np.float32)
    indices, counts, valid = pipeline.hist14_cells(points, schema)
    assert valid.tolist() == [True, True, True, True, False, False, False, False, False]
    assert counts.sum() == 4 and len(indices) == 3
    tensor = encode_hist14_chw(points, schema)
    reconstructed = pipeline.height_count_channels(indices, counts, schema)
    np.testing.assert_allclose(reconstructed, tensor[:4], atol=1e-7)


def test_feature_capture_uses_real_forward_and_removes_hooks_without_changing_state():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Sequential(torch.nn.Conv2d(2, 3, 1), torch.nn.ReLU())
        def forward(self, x):
            features = self.backbone(x)
            return {'cls': features, 'offset': features[:, :2]}
    model = Model().eval()
    tensor = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    expected = model(tensor)['cls'].detach().numpy()
    prediction, captured = pipeline.capture_forward(model, tensor, modules=['backbone.0', 'backbone'])
    np.testing.assert_array_equal(captured['backbone'], expected)
    np.testing.assert_array_equal(prediction['cls'].numpy(), expected)
    for module in model.modules():
        assert not module._forward_hooks
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_capture_cleans_up_hooks_after_failed_forward():
    class Broken(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Identity()
        def forward(self, x):
            self.backbone(x)
            raise RuntimeError('failed')
    model = Broken().eval()
    with pytest.raises(RuntimeError, match='failed'):
        pipeline.capture_forward(model, torch.ones(1, 1, 2, 2), modules=['backbone'])
    assert not model.backbone._forward_hooks


def test_feature_energy_retains_spatial_shape_and_actual_tensor_values():
    features = np.array([[[[3., 0.], [0., 8.]], [[4., 0.], [0., 6.]]]], np.float32)
    energy = pipeline.feature_energy(features)
    np.testing.assert_allclose(energy, [[np.sqrt(12.5), 0.], [0., np.sqrt(50.)]])
    assert energy.shape == (2, 2)
    with pytest.raises(ValueError):
        pipeline.feature_energy(np.ones((3, 4)))


def test_real_augmentor_is_reproducible_and_donor_db_excludes_scene_and_validation(tmp_path):
    from benchmark_fixtures import asset_fixture
    from tools.visualization.export_kitti_overview import DatasetPaths
    config, _, processed, raw = asset_fixture(tmp_path, box_mode='bev')
    paths = DatasetPaths(processed / 'pointcloud', raw / 'label_2', raw / 'calib')
    donor_root = tmp_path / 'preview_donors'
    report = pipeline.prepare_donors(paths, ['000000', '000001'], ['000002'], '000000',
                                     donor_root, config['data']['kitti']['objects'], donor_count=1)
    assert report['source_frame_ids'] == ['000001']
    assert report['scene_excluded'] and report['validation_excluded']
    assert all(count >= 1 for count in report['counts'].values())
    boxes, _ = pipeline.detector_boxes(raw / 'label_2/000000.txt', raw / 'calib/000000.txt',
                                       config['data']['kitti']['objects'])
    points = np.fromfile(processed / 'pointcloud/000000.bin', np.float32).reshape(-1, 4)
    before = points.copy()
    # Isolate world transforms with the real sampler queue's sampling probability disabled.
    config['augmentation']['AUG_CONFIG_LIST'][0]['PROBABILITY'] = 0.
    first = pipeline.augment_scene(points, boxes, config, donor_root, ['000000', '000001'], seed=2)
    second = pipeline.augment_scene(points, boxes, config, donor_root, ['000000', '000001'], seed=2)
    for left, right in zip(first[:3], second[:3]):
        np.testing.assert_array_equal(left, right)
    assert first[4]['probability_skipped'] is True
    assert not np.array_equal(first[1], points)
    np.testing.assert_array_equal(points, before)
    with pytest.raises(ValueError, match='train frame'):
        pipeline.prepare_donors(paths, ['000000', '000001'], ['000002'], '000002',
                                donor_root, config['data']['kitti']['objects'], donor_count=1)


def test_signed_feature_channels_are_displayed_with_symmetric_bounds(tmp_path):
    geometry = dict(x_min=0., x_max=4., y_min=-2., y_max=2.)
    ranges = pipeline.map_gallery([np.array([[-2., 1.], [0., 3.]])], ['feature channel 0'],
                                  geometry, tmp_path / 'signed', columns=1, feature=True)
    assert ranges[0]['display_min'] == -ranges[0]['display_max']
    assert (tmp_path / 'signed.png').is_file()
