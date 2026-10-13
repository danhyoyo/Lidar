"""Run this cell in Colab to export camera images matching the chosen KITTI bins."""
from pathlib import Path
import json
import zipfile

# current_clip: the 174 Object IDs used by our existing video, in temporal order.
# all_bins: every .bin in BIN_DIR, matched by filename within the selected split.
MODE = "current_clip"
SPLIT = "training"
ROOT = Path("/content")
BIN_DIR = ROOT / SPLIT / "velodyne"
CAMERAS = ("image_2", "image_3")

CLIP_IDS = """
002425 006794 006759 003318 000533 004864 006288 006786 004687 003259
000042 007262 004000 002619 004460 006698 003447 005262 006884 000881
001719 003689 006012 003365 001329 006636 000229 002749 006484 005555
004947 003750 000773 002025 003530 003869 004485 003583 003179 003878
001216 001863 001517 005297 004131 000792 002661 006643 000916 006533
005405 001318 006633 000589 005322 005062 005222 007391 000452 004697
003679 002356 001131 007164 007205 006561 005649 002919 004669 005375
000440 004256 004108 007292 000939 004792 001143 001658 000270 005827
001978 005926 004599 000235 006836 002866 003181 006954 003470 007247
001536 005923 006286 005807 002985 007119 004064 004124 002292 005642
000938 001296 007415 000506 005712 003300 001094 000376 007240 001755
002991 001145 004383 003302 000804 000035 003042 006623 002266 005213
005571 006785 005448 004850 005133 005528 006761 007016 003367 002690
006997 007157 000521 002625 000106 003058 004556 001133 003129 000569
004649 002556 000650 005656 005619 001786 002420 001263 000329 004173
005077 007385 006762 001148 000683 002702 006385 005325 001275 001542
007038 002021 000319 004891 002988 005334 000524 006999 001952 005081
001936 007083 002506 007030
""".split()


def export_images(root, split, ids, cameras, output):
    """Check every match before exporting, preserving split and camera paths."""
    root, output = Path(root), Path(output)
    ids, cameras = list(ids), tuple(cameras)
    if split not in ("training", "testing"):
        raise ValueError("SPLIT must be training or testing")
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Frame IDs must be nonempty and unique")
    if any(len(value) != 6 or not value.isascii() or not value.isdigit() for value in ids):
        raise ValueError("Each Object ID must contain six digits")
    if not cameras or len(cameras) != len(set(cameras)) or any(
        camera not in ("image_2", "image_3") for camera in cameras
    ):
        raise ValueError("CAMERAS must select image_2 and/or image_3")
    sources = [root / split / camera / f"{identifier}.png"
               for identifier in ids for camera in cameras]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images:\n" + "\n".join(missing[:20]))

    # PNG files are already compressed; store them without recompressing.
    partial = output.with_suffix(output.suffix + ".partial")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sources:
            archive.write(path, arcname=path.relative_to(root).as_posix())
        archive.writestr("frame_ids.txt", "\n".join(ids) + "\n")
        archive.writestr("manifest.json", json.dumps({
            "split": split, "cameras": cameras, "frame_count": len(ids),
            "image_count": len(sources), "object_ids": ids,
        }, indent=2))
    partial.replace(output)
    print(f"Exported {len(ids)} frames / {len(sources)} images: {output}")
    print(f"ZIP size: {output.stat().st_size / 1024**2:.1f} MiB")
    return output


def main():
    if MODE == "current_clip":
        if SPLIT != "training":
            raise ValueError("The current 174-frame clip belongs to training")
        ids = CLIP_IDS
    elif MODE == "all_bins":
        ids = sorted(path.stem for path in BIN_DIR.glob("*.bin"))
        if not ids:
            raise FileNotFoundError(f"No .bin files found in {BIN_DIR}")
    else:
        raise ValueError("MODE must be current_clip or all_bins")
    output = ROOT / f"kitti_{SPLIT}_{len(ids)}frames_images.zip"
    export_images(ROOT, SPLIT, ids, CAMERAS, output)
    from google.colab import files
    files.download(str(output))


if __name__ == "__main__":
    main()
