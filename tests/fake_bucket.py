"""In-memory stand-in for a GCS bucket (duck-typed like orchestration.bucket expects)."""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import google_crc32c


def crc32c_b64(data: bytes) -> str:
    return base64.b64encode(google_crc32c.Checksum(data).digest()).decode()


class FakeBlob:
    def __init__(self, bucket: "FakeBucket", name: str):
        self.bucket, self.name = bucket, name

    def upload_from_filename(self, path) -> None:
        self.bucket.uploads.append(self.name)
        self.bucket.objects[self.name] = Path(path).read_bytes()

    def download_to_filename(self, path) -> None:
        self.bucket.downloads.append(self.name)
        Path(path).write_bytes(self.bucket.objects[self.name])


class FakeBucket:
    def __init__(self, objects: dict[str, bytes] | None = None):
        self.objects = dict(objects or {})
        self.uploads: list[str] = []
        self.downloads: list[str] = []

    def list_blobs(self, prefix=None):
        return [SimpleNamespace(name=n, crc32c=crc32c_b64(b)) for n, b in sorted(self.objects.items())
                if prefix is None or n.startswith(prefix)]

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self, name)
