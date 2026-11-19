"""Tests for the helpers the GitHub workflows call: changed kernels, maintainers, publish target."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import changed_kernels  # noqa: E402
import kernel_maintainers  # noqa: E402
import publish_target  # noqa: E402


def add_kernel(root: Path, name: str, version: int = 1, org: str = "rocm-community"):
    k = root / name
    k.mkdir(parents=True)
    (k / "flake.nix").write_text("{}\n")
    (k / "build.toml").write_text(
        f'[general]\nname = "{name}"\nversion = {version}\n\n'
        f'[general.hub]\nrepo-id = "{org}/{name}"\n'
    )
    return k


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def test_changed_kernels(tmp_path):
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.email", "ci@example.com")
    git(tmp_path, "config", "user.name", "ci")
    add_kernel(tmp_path, "alpha")
    add_kernel(tmp_path, "beta")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "x.md").write_text("x\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "base")
    base = git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "alpha" / "kernel.hip").write_text("// change\n")
    (tmp_path / "docs" / "x.md").write_text("y\n")
    (tmp_path / "README.md").write_text("top-level files are not kernels\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "change")
    files = changed_kernels.changed_files(base, "HEAD", tmp_path)
    assert changed_kernels.kernels_in(files, tmp_path) == ["alpha"]
    assert changed_kernels.all_kernels(tmp_path) == ["alpha", "beta"]


def test_explicit_kernels_must_exist(tmp_path, capsys):
    add_kernel(tmp_path, "alpha")
    root = ["--root", str(tmp_path)]
    assert changed_kernels.main([*root, "--kernels", "alpha"]) == 0
    assert json.loads(capsys.readouterr().out) == ["alpha"]
    assert changed_kernels.main([*root, "--kernels", "nope"]) == 1
    assert changed_kernels.main([*root, "--kernels", "$(evil)"]) == 1


def write_registry(root: Path, registry: dict):
    (root / ".github").mkdir(exist_ok=True)
    (root / ".github" / "kernel-maintainers.json").write_text(json.dumps(registry))


def test_maintainers_ok(tmp_path):
    add_kernel(tmp_path, "alpha")
    write_registry(tmp_path, {"alpha": ["octocat", "hubot"]})
    assert kernel_maintainers.check(tmp_path) == []


def test_maintainers_problems(tmp_path):
    add_kernel(tmp_path, "alpha")
    add_kernel(tmp_path, "beta")
    add_kernel(tmp_path, "gamma")
    write_registry(tmp_path, {"alpha": [], "beta": ["@octocat"], "gone": ["x"]})
    problems = "\n".join(kernel_maintainers.check(tmp_path))
    assert "alpha: needs a list of at least one GitHub handle" in problems
    assert "beta: not GitHub handles" in problems
    assert "gamma: missing from" in problems
    assert "gone: in .github/kernel-maintainers.json but there is no kernel" in problems


def test_publish_target_defaults(tmp_path):
    add_kernel(tmp_path, "alpha", version=1)
    t = publish_target.target(tmp_path, "alpha", "rocm-kernels-test")
    assert t["repo-id"] == "rocm-kernels-test/alpha"
    assert t["source-repo-id"] == "rocm-community/alpha"
    assert (t["branch"], t["also-main"], t["load-args"]) == (
        "v1",
        "true",
        "--version 1",
    )
    assert t["config-dir"] == "alpha"


def test_publish_target_branch(tmp_path):
    add_kernel(tmp_path, "alpha", version=2)
    t = publish_target.target(tmp_path, "alpha", "rocm-kernels-test", "staging")
    assert (t["branch"], t["also-main"], t["load-args"]) == (
        "staging",
        "false",
        "--revision staging",
    )


@pytest.mark.parametrize(
    "kernel,org,branch",
    [
        ("alpha", "amd", ""),  # reserved for AMD's packages
        ("alpha", "", ""),  # HUB_ORG not set
        ("alpha", "test-org", "bad branch"),
        ("../alpha", "test-org", ""),
    ],
)
def test_publish_target_rejects(tmp_path, kernel, org, branch):
    add_kernel(tmp_path, "alpha")
    with pytest.raises(ValueError):
        publish_target.target(tmp_path, kernel, org, branch)


def test_every_workflow_upload_is_private():
    """kernel-builder creates public repositories unless told otherwise."""
    workflows = Path(__file__).resolve().parents[2] / ".github" / "workflows"
    uploads = []
    for wf in sorted(workflows.glob("*.yml")):
        # Join shell continuation lines, so each command is one line.
        text = re.sub(r"\\\n\s*", " ", wf.read_text())
        uploads += [
            (wf.name, line)
            for line in text.splitlines()
            if re.search(r"kernel-builder\S*\s+--\s+upload\b", line)
        ]
    assert uploads, "no kernel-builder upload found in the workflows"
    for name, line in uploads:
        assert "--private" in line.split(), f"{name}: upload without --private: {line}"


def job_blocks(text: str) -> dict[str, str]:
    """Each job's YAML text, by job id (jobs are indented two spaces under `jobs:`)."""
    body = text.split("\njobs:\n", 1)[-1]
    parts = re.split(r"^  ([A-Za-z0-9_-]+):\n", body, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2]))


def test_gpu_jobs_run_in_the_ci_image():
    """Jobs on the MI350 runners run in a CI image, with the GPU devices passed in."""
    workflows = Path(__file__).resolve().parents[2] / ".github" / "workflows"
    gpu_jobs = 0
    for wf in sorted(workflows.glob("*.yml")):
        for job, block in job_blocks(wf.read_text()).items():
            if not re.search(r"^    runs-on: .*\blinux-rck-mi350-\d", block, re.M):
                continue
            gpu_jobs += 1
            where = f"{wf.name}: {job}"
            has_container = re.search(r"^    container:\n", block, re.M)
            assert has_container, f"{where}: no container"
            image = r"^      image: \$\{\{ vars\.(CI_IMAGE|CI_USER_IMAGE) "
            assert re.search(image, block, re.M), f"{where}: image not from a variable"
            options = re.search(r"^      options: (.*)$", block, re.M)
            assert options, f"{where}: no container options"
            for device in ("/dev/kfd", "/dev/dri"):
                assert f"--device {device}" in options.group(1), f"{where}: no {device}"
    assert gpu_jobs >= 2  # rocm-build-test.yml build-test, publish.yml verify
