import pytest

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


def test_push_skips_logs_and_tmp(tmp_path):
    fake = FakeBucket()
    _write(tmp_path, "logs/refresh.log", "x")
    _write(tmp_path, "processed/a.json.tmp", "x")
    _write(tmp_path, "processed/a.json", "1")
    assert sync.push(fake, tmp_path) == ["processed/a.json"]


def test_pull_skips_folder_markers(tmp_path):
    fake = FakeBucket({"processed/": b"", "processed/a.json": b"a"})
    assert sync.pull(fake, tmp_path) == ["processed/a.json"]


def test_pull_failed_download_leaves_no_tmp(tmp_path):
    class Broken(FakeBucket):
        def blob(self, name):
            blob = super().blob(name)

            def fail(path):
                open(path, "wb").write(b"part")
                raise OSError("network")
            blob.download_to_filename = fail
            return blob
    with pytest.raises(OSError):
        sync.pull(Broken({"processed/a.json": b"a"}), tmp_path)
    assert not list(tmp_path.rglob("*.tmp")) and not (tmp_path / "processed/a.json").exists()


def test_main_reads_bucket_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("FPL_COACH_BUCKET", raising=False)
    seen = []
    monkeypatch.setattr(sync, "load_env", lambda: monkeypatch.setenv("FPL_COACH_BUCKET", "b"))
    monkeypatch.setattr(sync, "open_bucket", lambda name: seen.append(name) or FakeBucket())
    monkeypatch.setattr("sys.argv", ["bucket", "push", "--data-dir", str(tmp_path)])
    sync.main()
    assert seen == ["b"]
