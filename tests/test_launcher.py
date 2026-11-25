import pytest

from presentation.api import launcher


def test_cloudrun_passes_args():
    class Client:
        def run_job(self, request):
            self.request = request

    c = Client()
    launcher.CloudRunLauncher("projects/p/locations/l/jobs/coach", client=c).start(42, 7)
    assert c.request.name == "projects/p/locations/l/jobs/coach"
    assert list(c.request.overrides.container_overrides[0].args) == ["coach", "--entry", "42", "--run-id", "7"]


def test_local_launcher_spawns_job(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda cmd, **kw: seen.update(cmd=cmd, kw=kw))
    launcher.LocalLauncher(tmp_path).start(42, 7)
    assert seen["cmd"][1:] == ["-m", "orchestration.job", "coach", "--entry", "42", "--run-id", "7"]
    assert (tmp_path / "logs" / "run-7.log").exists()


def test_cloudrun_needs_job_name(monkeypatch, tmp_path):
    monkeypatch.setenv("LAUNCHER", "cloudrun")
    monkeypatch.delenv("COACH_JOB", raising=False)
    with pytest.raises(ValueError):
        launcher.launcher_from_env(tmp_path)
