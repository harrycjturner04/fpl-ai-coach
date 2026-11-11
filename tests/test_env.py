from ingestion.env import load_env


def test_load_env_parses_and_keeps_existing(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text('# comment\n\nFPL_A=one\nFPL_B = "two"\nFPL_C=x=y\nFPL_KEEP=file\nFPL_BLANK=\nnot a pair\n', encoding="utf-8")
    for name in ("FPL_A", "FPL_B", "FPL_C", "FPL_BLANK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FPL_KEEP", "shell")
    load_env(f)
    import os
    assert (os.environ["FPL_A"], os.environ["FPL_B"], os.environ["FPL_C"]) == ("one", "two", "x=y")
    assert os.environ["FPL_KEEP"] == "shell"
    assert "FPL_BLANK" not in os.environ


def test_load_env_missing_file_is_noop(tmp_path):
    load_env(tmp_path / "absent.env")
