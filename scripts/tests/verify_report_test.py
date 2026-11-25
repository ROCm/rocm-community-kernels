"""verify_report.py: the verify job summary, and stage.yml's status and comment."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import verify_report  # noqa: E402

PASSED = {
    "repo_id": "staging/scale-triton",
    "ok": True,
    "variant": "torch210-cxx11-rocm71-x86_64-linux",
    "torch": "2.10.0+rocm7.1",
    "hip": "7.1",
    "kernels": "0.19.0",
    "tests_exit": 0,
    "warnings": [],
}
ALL_JOBS = {j: "success" for j in verify_report.JOBS}


def stage(jobs=None, reports=None):
    return verify_report.stage(
        "scale-triton",
        "0123456789abcdef" * 2 + "01234567",
        "https://github.com/o/r/actions/runs/1",
        "staging/scale-triton",
        "pr-11",
        ALL_JOBS if jobs is None else jobs,
        {"gfx950": PASSED} if reports is None else reports,
    )


def test_summary():
    text = verify_report.summary(PASSED)
    assert text.startswith(
        "### get_kernel(`staging/scale-triton`) from the Hub: passed"
    )
    assert "- **variant**: `torch210-cxx11-rocm71-x86_64-linux`" in text
    assert "FAILED" in verify_report.summary({**PASSED, "ok": False, "error": "x"})


def test_stage_passed():
    state, description, body = stage()
    assert state == "success"
    assert description == "get_kernel() and kernels_ci tests passed: gfx950"
    assert body.startswith("<!-- hub-staging:scale-triton -->\n")
    assert "| `gfx950` | passed | `torch210-cxx11-rocm71-x86_64-linux`" in body
    assert "Publish kernel" in body


def test_stage_failed_job():
    state, description, _ = stage(jobs={**ALL_JOBS, "build": "failure"})
    assert state == "failure" and description == "failed at build (failure)"
    state, description, _ = stage(jobs={**ALL_JOBS, "upload": None})
    assert description == "failed at upload (not run)"


def test_stage_failed_or_missing_report():
    state, description, body = stage(reports={"gfx950": {**PASSED, "ok": False}})
    assert state == "failure" and "gfx950" in description
    state, _, body = stage(reports={"gfx950": None})
    assert state == "failure" and "no report" in body
    assert stage(reports={})[0] == "failure"


def test_report_text_cannot_add_markup():
    """The report comes from a job that ran the pull request's tests."""
    evil = "x`</td>| @everyone [link](https://evil)\n## heading" + "y" * 500
    _, description, body = stage(
        reports={"gfx950": {**PASSED, "ok": False, "variant": evil, "error": evil}}
    )
    assert "\n## heading" not in body
    for line in body.splitlines():
        if "@everyone" in line:
            # only ever inside a code span, with its pipes escaped
            assert "`x'</td>\\| @everyone" in line
    assert max(len(line) for line in body.splitlines()) < 1000
    assert len(description) <= 140


def test_cli(tmp_path):
    reports = tmp_path / "reports"
    (reports / "verify-scale-triton-gfx950").mkdir(parents=True)
    (reports / "verify-scale-triton-gfx950" / "verify.json").write_text(
        json.dumps(PASSED)
    )
    needs = {j: {"result": "success", "outputs": {}} for j in verify_report.JOBS}
    args = [
        "stage",
        "--kernel",
        "scale-triton",
        "--sha",
        "a" * 40,
        "--run-url",
        "https://example.com/run",
        "--repo-id",
        "staging/scale-triton",
        "--branch",
        "pr-11",
        "--gpu-jobs",
        json.dumps([{"kernel": "scale-triton", "arch": "gfx950", "runner": "r"}]),
        "--jobs",
        json.dumps(needs),
        "--reports",
        str(reports),
        "--status-json",
        str(tmp_path / "status.json"),
        "--body",
        str(tmp_path / "body.md"),
    ]
    assert verify_report.main(args) == 0
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["state"] == "success" and status["context"] == "Hub staging"
    assert "passed" in (tmp_path / "body.md").read_text()


def test_summary_cli_without_report(tmp_path, capsys):
    assert verify_report.main(["summary", str(tmp_path / "missing.json")]) == 0
    assert "No verification report" in capsys.readouterr().out
