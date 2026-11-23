"""ci_download.py with a stub huggingface_hub: what it asks the Hub for, and its report."""

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ci_download  # noqa: E402


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """A stub huggingface_hub whose snapshot_download records its arguments."""
    calls = []

    def snapshot_download(repo_id, **kw):
        calls.append({"repo_id": repo_id, **kw})
        snapshot = Path(kw["cache_dir"]) / "snapshots" / ("c" * 40)
        variant = snapshot / "build" / "torch-rocm"
        variant.mkdir(parents=True, exist_ok=True)
        (variant / "metadata.json").write_text("{}")
        return str(snapshot)

    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setenv("KERNELS_CACHE", str(tmp_path / "cache"))
    return calls


def test_downloads_the_branch_into_kernels_cache(hub, tmp_path):
    report = tmp_path / "download.json"
    rc = ci_download.main(
        ["--repo-id", "test-org/demo", "--revision", "v1", "--json", str(report)]
    )
    assert rc == 0
    (call,) = hub
    assert call["repo_id"] == "test-org/demo"
    assert call["repo_type"] == "kernel"
    assert call["revision"] == "v1"
    assert call["cache_dir"] == str(tmp_path / "cache")
    assert call["allow_patterns"] == "build/*"
    r = json.loads(report.read_text())
    assert r["ok"] and r["commit"] == "c" * 40 and r["variants"] == ["torch-rocm"]


def test_needs_kernels_cache(hub, monkeypatch, tmp_path):
    monkeypatch.delenv("KERNELS_CACHE")
    report = tmp_path / "download.json"
    rc = ci_download.main(
        ["--repo-id", "test-org/demo", "--revision", "v1", "--json", str(report)]
    )
    assert rc == 1 and not hub
    assert "KERNELS_CACHE" in json.loads(report.read_text())["error"]


def test_report_never_holds_the_token(hub, monkeypatch, tmp_path):
    # Built at run time, so that no token-shaped string is in the repository.
    token = "hf_" + "NotARealToken" * 3
    monkeypatch.setenv("HF_TOKEN", token)
    report = tmp_path / "download.json"
    ci_download.main(
        ["--repo-id", "test-org/demo", "--revision", "v1", "--json", str(report)]
    )
    assert token not in report.read_text()
