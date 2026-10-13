"""Verify real RGB pairing and partial ZIP extraction without network fixtures."""
import importlib
import io
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def downloader():
    return importlib.import_module("tools.visualization.download_kitti_sequence_images")


def exporter():
    return importlib.import_module("tools.visualization.export_kitti_camera_video")


@pytest.mark.parametrize("compression", [ZIP_STORED, ZIP_DEFLATED])
def test_selected_zip_member_is_extracted_exactly_and_crc_checked(compression):
    data = io.BytesIO()
    payload = bytes(range(256)) * 100
    with ZipFile(data, "w", compression=compression) as z:
        z.writestr("training/image_2/000123.png", payload)
    raw = data.getvalue()
    with ZipFile(io.BytesIO(raw)) as z:
        info = z.getinfo("training/image_2/000123.png")
    ranges = []
    def fetch(start, length):
        ranges.append((start, length))
        return raw[start:start + length]
    extracted = downloader().read_zip_member(info, fetch)
    assert extracted == payload
    assert len(ranges) == 2
    def corrupted(start, length):
        chunk = bytearray(raw[start:start + length])
        if start > info.header_offset:
            chunk[-1] ^= 255
        return bytes(chunk)
    with pytest.raises(ValueError):
        downloader().read_zip_member(info, corrupted)


def test_camera_paths_use_object_id_not_video_index_or_raw_frame(tmp_path):
    (tmp_path / "002425.png").write_bytes(b"fixture")
    entries = [{"object_id": "002425", "raw_frame": 77, "video_index": 0}]
    assert exporter().camera_paths(entries, tmp_path) == [tmp_path / "002425.png"]
    (tmp_path / "002425.png").unlink()
    (tmp_path / "000077.png").write_bytes(b"wrong frame")
    with pytest.raises(FileNotFoundError):
        exporter().camera_paths(entries, tmp_path)


def test_camera_renderer_keeps_source_rgb_and_preserves_image_aspect(tmp_path):
    from tools.visualization.export_kitti_sequence_video import SequenceRenderer
    renderer = SequenceRenderer(roi=[0, 8, -4, 4, -2.5, 3], width=960, height=540,
        elevation=28, azimuth=-145, heatmap_extent=[0, 8, -4, 4],
        title="Camera fixture", score_threshold=.3, camera_image_size=(1242, 375))
    rgb = np.zeros((375, 1242, 3), dtype=np.uint8)
    rgb[25:100, 100:200] = [11, 22, 33]
    renderer.render(np.array([[1, 1, -1, .5]], dtype=np.float32), np.empty((0, 7)),
        np.zeros((3, 20, 20), dtype=np.float32), class_ids={"Car": 0, "Pedestrian": 1, "Cyclist": 2},
        index=0, total=1, entry={"object_id": "002425", "raw_frame": 77}, fps=10,
        camera_image=rgb, output=tmp_path / "camera.png")
    np.testing.assert_array_equal(renderer.camera_artist.get_array(), rgb)
    assert renderer.camera_artist.axes.get_aspect() == 1
    with Image.open(tmp_path / "camera.png") as image:
        assert image.size == (960, 540)
    renderer.close()


def test_colab_zip_extracts_exact_object_pairs_and_rejects_wrong_split(tmp_path):
    payload = io.BytesIO()
    Image.new("RGB", (12, 4), (20, 30, 40)).save(payload, format="PNG")
    entries = [{"object_id": "002425"}, {"object_id": "006794"}]
    source = tmp_path / "images.zip"
    with ZipFile(source, "w") as archive:
        archive.writestr("frame_ids.txt", "002425\n006794\n")
        for camera in ("image_2", "image_3"):
            for entry in entries:
                archive.writestr(f"training/{camera}/{entry['object_id']}.png", payload.getvalue())
        archive.writestr("../unexpected.txt", "must never be extracted")
    destination = tmp_path / "images"
    report = exporter().extract_camera_zip(source, destination, entries)
    assert report["image_count"] == 4
    assert report["image_size"] == [12, 4]
    assert len(list(destination.rglob("*.png"))) == 4
    assert not (tmp_path / "unexpected.txt").exists()
    wrong = tmp_path / "testing.zip"
    with ZipFile(wrong, "w") as archive:
        for entry in entries:
            archive.writestr(f"testing/image_2/{entry['object_id']}.png", payload.getvalue())
    with pytest.raises(FileNotFoundError):
        exporter().extract_camera_zip(wrong, tmp_path / "wrong", entries)
    assert not (tmp_path / "wrong").exists()


def test_stereo_renderer_does_not_swap_images_or_reuse_previous_frame(tmp_path):
    from tools.visualization.export_kitti_sequence_video import SequenceRenderer
    renderer = SequenceRenderer(roi=[0, 8, -4, 4, -2.5, 3], width=960, height=540,
        elevation=28, azimuth=-145, heatmap_extent=[0, 8, -4, 4], title="Stereo",
        score_threshold=.3, camera_image_size=(12, 4), right_camera_image_size=(12, 4))
    left = np.full((4, 12, 3), [10, 20, 30], dtype=np.uint8)
    right = np.full((4, 12, 3), [40, 50, 60], dtype=np.uint8)
    arguments = dict(class_ids={"Car": 0, "Pedestrian": 1, "Cyclist": 2},
        total=2, entry={"object_id": "002425", "raw_frame": 77}, fps=10,
        output=tmp_path / "stereo.png")
    for index in range(2):
        renderer.render(np.empty((0, 4)), np.empty((0, 7)), np.zeros((3, 20, 20)),
            index=index, camera_image=left + index, right_camera_image=right + index, **arguments)
        np.testing.assert_array_equal(renderer.camera_artist.get_array(), left + index)
        np.testing.assert_array_equal(renderer.right_camera_artist.get_array(), right + index)
    with pytest.raises(ValueError, match="RGB"):
        renderer.render(np.empty((0, 4)), np.empty((0, 7)), np.zeros((3, 20, 20)),
            index=1, camera_image=left, **arguments)
    renderer.close()
