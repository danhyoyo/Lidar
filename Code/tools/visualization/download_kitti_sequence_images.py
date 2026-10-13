#!/usr/bin/env python3
"""Fetch only selected training/image_2 PNG members from the official KITTI ZIP."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile, ZIP_STORED, ZIP_DEFLATED
import zlib

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"


class RemoteZIP(io.RawIOBase):
    """Seekable ZIP directory reader; reject full downloads and bounded range mismatches."""

    def __init__(self, url):
        self.url, self.position, self.etag = url, 0, None
        data, start, end, total, etag = self._request("bytes=-131072")
        self.length, self.etag = total, etag
        self.tail_start, self.tail = start, data

    def _request(self, range_header):
        headers = {"Range": range_header, "User-Agent": "KITTI-selected-frame-visualization/1.0"}
        if self.etag:
            headers["If-Match"] = self.etag
        for attempt in range(3):
            try:
                with urlopen(Request(self.url, headers=headers), timeout=30) as response:
                    if response.status != 206:
                        raise ValueError("Server did not honour HTTP Range; refusing a full archive download")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match:
                        raise ValueError("Missing or malformed HTTP Content-Range")
                    start, end, total = map(int, match.groups())
                    expected = end - start + 1
                    if not 0 < expected <= 16 * 1024 * 1024:
                        raise ValueError("Refusing an unexpectedly large archive range")
                    data = response.read(expected + 1)
                    if len(data) != expected:
                        raise ValueError("Truncated or oversized HTTP range response")
                    return data, start, end, total, response.headers.get("ETag")
            except (URLError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    def fetch(self, start, length):
        if length == 0:
            return b""
        if start < 0 or start + length > self.length:
            raise ValueError("ZIP range outside source archive")
        if start >= self.tail_start:
            offset = start - self.tail_start
            return self.tail[offset:offset + length]
        data, actual_start, actual_end, total, _ = self._request(f"bytes={start}-{start + length - 1}")
        if (actual_start, actual_end, total) != (start, start + length - 1, self.length):
            raise ValueError("Server returned a different archive byte range")
        return data

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.length)
        if self.position < 0:
            raise ValueError("Negative ZIP seek")
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        size = self.length - self.position if size < 0 else min(size, self.length - self.position)
        if size > 16 * 1024 * 1024:
            raise ValueError("Refusing an unbounded archive read")
        result = self.fetch(self.position, size)
        self.position += len(result)
        return result


def read_zip_member(info, fetch):
    """Extract a stored/deflated member with two precise byte reads and CRC verification."""
    header = fetch(info.header_offset, 30)
    if len(header) != 30:
        raise ValueError("Truncated ZIP local header")
    fields = struct.unpack("<4s5H3I2H", header)
    if fields[0] != b"PK\x03\x04" or fields[2] & 1 or fields[3] != info.compress_type:
        raise ValueError("Unsupported or inconsistent ZIP local header")
    offset = info.header_offset + 30 + fields[-2] + fields[-1]
    compressed = fetch(offset, info.compress_size)
    if len(compressed) != info.compress_size:
        raise ValueError("Truncated ZIP member payload")
    if info.compress_type == ZIP_STORED:
        payload = compressed
    elif info.compress_type == ZIP_DEFLATED:
        try:
            payload = zlib.decompress(compressed, -15)
        except zlib.error as error:
            raise ValueError("Corrupt deflated ZIP member") from error
    else:
        raise ValueError(f"Unsupported ZIP compression: {info.compress_type}")
    if len(payload) != info.file_size or zlib.crc32(payload) & 0xffffffff != info.CRC:
        raise ValueError("KITTI image ZIP member failed size/CRC verification")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-url", default=DEFAULT_URL)
    parser.add_argument("--sequence-dir", type=Path, default=ROOT.parent / "Result/artifacts/kitti_sequence_video")
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "Result/artifacts/kitti_sequence_images")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("workers must be positive")
    from PIL import Image
    manifest_path = args.sequence_dir / "frame_manifest.json"
    entries = json.loads(manifest_path.read_text())
    identifiers = [entry["object_id"] for entry in entries]
    if (not identifiers or len(set(identifiers)) != len(identifiers)
            or any(not re.fullmatch(r"\d{6}", value) for value in identifiers)):
        raise ValueError("Expected a sequence manifest with distinct six-digit Object IDs")
    output = args.output_dir.expanduser().resolve()
    image_dir = output / "image_2"
    image_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading official ZIP directory; fetching only {len(entries)} left colour images", flush=True)
    remote = RemoteZIP(args.source_url)
    with ZipFile(remote) as archive:
        infos = {identifier: archive.getinfo(f"training/image_2/{identifier}.png") for identifier in identifiers}
    estimated = sum(info.compress_size for info in infos.values())
    print(f"Selected image payload: {estimated / 1024**2:.1f} MiB; archive stays remote", flush=True)
    def download(identifier):
        info = infos[identifier]
        path = image_dir / f"{identifier}.png"
        payload = path.read_bytes() if path.exists() else None
        cached = payload is not None and len(payload) == info.file_size and zlib.crc32(payload) & 0xffffffff == info.CRC
        if not cached:
            payload = read_zip_member(info, remote.fetch)
        with Image.open(io.BytesIO(payload)) as image:
            if image.format != "PNG":
                raise ValueError(f"Expected a real PNG member: {info.filename}")
            size = image.size
            image.verify()
        if not cached:
            temporary = path.with_suffix(".png.partial")
            temporary.write_bytes(payload)
            temporary.replace(path)
        return {"object_id": identifier, "zip_member": info.filename, "path": str(path),
                "sha256": hashlib.sha256(payload).hexdigest(), "size_wh": list(size),
                "zip_crc32": info.CRC, "zip_header_offset": info.header_offset,
                "compressed_bytes": info.compress_size, "file_bytes": info.file_size,
                "cached_crc_verified": bool(cached)}
    results = {}
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download, identifier): identifier for identifier in identifiers}
        for future in as_completed(futures):
            result = future.result()
            results[result["object_id"]] = result
            if len(results) == 1 or len(results) % 10 == 0 or len(results) == len(entries):
                print(f"{len(results)}/{len(entries)} images CRC verified; {time.perf_counter() - started:.1f}s", flush=True)
    metadata = {"source_url": args.source_url, "archive_etag": remote.etag,
        "archive_size_bytes": remote.length, "camera": "left colour image_2",
        "image_count": len(results), "selected_compressed_bytes": estimated,
        "sequence_manifest": str(manifest_path.resolve()),
        "sequence_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "images": [results[identifier] for identifier in identifiers],
        "reproduce_command": [sys.executable, *sys.argv]}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved exact source PNGs to {image_dir}", flush=True)


if __name__ == "__main__":
    main()
