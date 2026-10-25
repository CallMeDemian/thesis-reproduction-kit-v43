import io
import os
import subprocess
import sys

from thesis_repro import cli

KOREAN = "손익계산서.xlsx"


def test_utf8_stdio_reconfigures_cp1252_streams(monkeypatch):
    out = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    err = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    cli._utf8_stdio()
    print(KOREAN)
    print(KOREAN, file=sys.stderr)
    for stream in (out, err):
        stream.flush()
        assert stream.encoding == "utf-8"
        assert stream.buffer.getvalue().decode("utf-8").strip() == KOREAN


def test_utf8_stdio_skips_streams_without_reconfigure(monkeypatch):
    monkeypatch.setattr(sys, "stdout", object())
    cli._utf8_stdio()


def test_korean_prints_on_cp1252_console():
    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    env.pop("PYTHONUTF8", None)
    code = f"from thesis_repro.cli import _utf8_stdio; _utf8_stdio(); print({KOREAN!r})"
    completed = subprocess.run([sys.executable, "-X", "utf8=0", "-c", code], env=env, capture_output=True)
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert completed.stdout.decode("utf-8").strip() == KOREAN
