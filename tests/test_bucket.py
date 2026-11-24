from orchestration import bucket as sync
from tests.fake_bucket import FakeBucket


def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_push_uploads_new_and_changed_only(tmp_path):
    fake = FakeBucket()
    _write(tmp_path, "processed/a.json", "1")
    _write(tmp_path, "processed/b.json", "1")
    assert sorted(sync.push(fake, tmp_path)) == ["processed/a.json", "processed/b.json"]
    _write(tmp_path, "processed/b.json", "2")
    assert sync.push(fake, tmp_path) == ["processed/b.json"]


def test_push_skips_raw(tmp_path):
    fake = FakeBucket()
    _write(tmp_path, "raw/x/bootstrap-static.json", "{}")
    _write(tmp_path, "processed/a.json", "1")
    assert sync.push(fake, tmp_path) == ["processed/a.json"]
    assert "raw/x/bootstrap-static.json" not in fake.objects


def test_push_uploads_metadata_and_snapshots_last(tmp_path):
    fake = FakeBucket()
    for rel in ["reports/1_gw1.json", "processed/metadata.json", "processed/z.parquet", "processed/a.json"]:
        _write(tmp_path, rel, "x")
    sync.push(fake, tmp_path)
    assert fake.uploads[-2:] == ["processed/metadata.json", "reports/1_gw1.json"]


def test_pull_replaces_changed_and_keeps_local_only(tmp_path):
    fake = FakeBucket({"processed/a.json": b"new", "processed/same.json": b"s"})
    _write(tmp_path, "processed/a.json", "old")
    _write(tmp_path, "processed/same.json", "s")
    _write(tmp_path, "processed/local.json", "mine")
    assert sync.pull(fake, tmp_path) == ["processed/a.json"]
    assert (tmp_path / "processed/a.json").read_text() == "new"
    assert (tmp_path / "processed/local.json").read_text() == "mine"
    assert not list(tmp_path.rglob("*.tmp"))


def test_pull_only_limits_paths(tmp_path):
    fake = FakeBucket({"processed/metadata.json": b"m", "processed/a.json": b"a"})
    assert sync.pull(fake, tmp_path, only=["processed/metadata.json"]) == ["processed/metadata.json"]
    assert not (tmp_path / "processed/a.json").exists()
