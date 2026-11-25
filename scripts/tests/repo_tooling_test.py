"""Tests for the helpers the GitHub workflows call (changed kernels, maintainers,
publish target and gate), and for the workflows themselves."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import changed_kernels  # noqa: E402
import check_wrapping  # noqa: E402
import kernel_maintainers  # noqa: E402
import publish_gate  # noqa: E402
import publish_target  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def add_kernel(
    root: Path,
    name: str,
    version: int = 1,
    org: str = "rocm-community",
    archs: tuple[str, ...] = ("gfx950",),
):
    k = root / name
    k.mkdir(parents=True)
    (k / "flake.nix").write_text("{}\n")
    (k / "build.toml").write_text(
        f'[general]\nname = "{name}"\nversion = {version}\n\n'
        f'[general.hub]\nrepo-id = "{org}/{name}"\n\n'
        f"[torch-noarch]\nrocm_archs = {json.dumps(list(archs))}\n"
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
    runner = check_wrapping.load_community_rules()["ci-runners"]["gfx950"]
    assert json.loads(t["gpu-jobs"]) == [
        {"kernel": "alpha", "arch": "gfx950", "runner": runner}
    ]
    assert t["build-arch"] == "gfx950"


def test_publish_target_needs_a_ci_gpu(tmp_path):
    add_kernel(tmp_path, "alpha", archs=("gfx942",))
    with pytest.raises(ValueError, match="CI runner"):
        publish_target.target(tmp_path, "alpha", "rocm-kernels-test")


RUNNERS = {"gfx950": "runner-mi350", "gfx942": "runner-mi300"}


def test_gpu_jobs_follow_the_supported_gpus(tmp_path):
    """Each kernel is tested on every GPU it lists that CI has a runner for, only."""
    add_kernel(tmp_path, "both", archs=("gfx942", "gfx950"))
    add_kernel(tmp_path, "mi300", archs=("gfx942", "gfx1201"))
    add_kernel(tmp_path, "nogpu", archs=("gfx1201",))
    jobs = changed_kernels.gpu_jobs(tmp_path, ["both", "mi300", "nogpu"], RUNNERS)
    assert [(j["kernel"], j["arch"], j["runner"]) for j in jobs] == [
        ("both", "gfx942", "runner-mi300"),
        ("both", "gfx950", "runner-mi350"),
        ("mi300", "gfx942", "runner-mi300"),
    ]


def test_gpu_jobs_reject_unsafe_runners(tmp_path):
    add_kernel(tmp_path, "alpha")
    with pytest.raises(ValueError):
        changed_kernels.gpu_jobs(tmp_path, ["alpha"], {"gfx950": "x; rm -rf /"})


def test_changed_kernels_github_output(tmp_path, monkeypatch, capsys):
    add_kernel(tmp_path, "alpha")
    out = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    root = ["--root", str(tmp_path)]
    assert changed_kernels.main([*root, "--kernels", "alpha", "--github-output"]) == 0
    lines = dict(ln.split("=", 1) for ln in out.read_text().splitlines())
    assert json.loads(lines["kernels"]) == ["alpha"]
    assert [j["arch"] for j in json.loads(lines["gpu-jobs"])] == ["gfx950"]


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


PROTECTED = {
    "name": "hf-rocm-community",
    "protection_rules": [
        {"type": "required_reviewers", "reviewers": [{"type": "Team"}]},
        {"type": "branch_policy"},
    ],
    "deployment_branch_policy": {
        "protected_branches": False,
        "custom_branch_policies": True,
    },
}


def test_publish_gate_ok():
    assert publish_gate.problems(PROTECTED, write_token_visible=False) == []


@pytest.mark.parametrize(
    "environment, visible, expected",
    [
        (None, False, "doesn't exist"),
        (PROTECTED, True, "HF_TOKEN is visible"),
        ({**PROTECTED, "protection_rules": []}, False, "no required reviewers"),
        (
            {
                **PROTECTED,
                "protection_rules": [{"type": "required_reviewers", "reviewers": []}],
            },
            False,
            "no required reviewers",
        ),
        ({**PROTECTED, "deployment_branch_policy": None}, False, "any branch"),
    ],
)
def test_publish_gate_problems(environment, visible, expected):
    found = publish_gate.problems(environment, write_token_visible=visible)
    assert len(found) == 1 and expected in found[0], found


def test_publish_gate_cli(tmp_path, capsys):
    env_json = tmp_path / "environment.json"
    args = ["--environment-json", str(env_json), "--write-token-visible", "false"]
    assert publish_gate.main(args) == 1  # missing file: no environment
    env_json.write_text(json.dumps(PROTECTED))
    assert publish_gate.main(args) == 0
    assert publish_gate.main(args[:3] + ["true"]) == 1


def test_publish_gate_checks_the_upload_environment():
    """The gate reads the environment the upload job waits in."""
    text = (WORKFLOWS / "publish.yml").read_text()
    upload = job_blocks(text)["upload"]
    name = re.search(r"^    environment:\n      name: (\S+)$", upload, re.M)
    assert name and name.group(1) == publish_gate.ENVIRONMENT
    assert f"/environments/{publish_gate.ENVIRONMENT}" in job_blocks(text)["target"]


def test_publish_gate_other_environment(tmp_path):
    """stage.yml checks hf-staging and HF_STAGING_TOKEN with the same rules."""
    found = publish_gate.problems(None, True, "hf-staging", "HF_STAGING_TOKEN")
    assert "HF_STAGING_TOKEN is visible" in found[0] and "hf-staging" in found[1]
    env_json = tmp_path / "environment.json"
    env_json.write_text(json.dumps(PROTECTED))
    args = ["--environment-json", str(env_json), "--write-token-visible", "false"]
    assert publish_gate.main(args + ["--environment", "hf-staging"]) == 0


def test_publish_target_runners_come_from_the_trusted_config(tmp_path):
    """stage.yml reads a pull request's kernel with --root: a ci/community.toml in
    that tree must not choose the runners."""
    add_kernel(tmp_path, "alpha")
    (tmp_path / "ci").mkdir()
    (tmp_path / "ci" / "community.toml").write_text(
        '[ci-runners]\ngfx950 = "attacker-runner"\n'
    )
    t = publish_target.target(tmp_path, "alpha", "staging-org", "pr-7")
    trusted = check_wrapping.load_community_rules()["ci-runners"]["gfx950"]
    assert [j["runner"] for j in json.loads(t["gpu-jobs"])] == [trusted]
    assert t["branch"] == "pr-7" and t["load-args"] == "--revision pr-7"
    assert t["also-main"] == "false"


def workflow_triggers(text: str) -> list[str]:
    on = re.search(r"^on:\n((?:  .*\n|\n)*)", text, re.M)
    return re.findall(r"^  ([a-z_]+):", on.group(1), re.M) if on else []


def test_stage_is_started_by_hand_from_main():
    """Only someone with write access can stage (workflow_dispatch), never a pull
    request or a comment, and only main's reviewed workflow and scripts run it."""
    text = (WORKFLOWS / "stage.yml").read_text()
    assert workflow_triggers(text) == ["workflow_dispatch"]
    target = job_blocks(text)["target"]
    assert "if: github.ref != 'refs/heads/main'" in target
    assert "exit 1" in target.split("Stage from main only", 1)[1].split("- name:")[0]


def test_stage_tokens_and_permissions():
    """The staging write token is used only in the upload job, which waits in the
    hf-staging environment; the read token only in the download job; only target
    and report can set statuses, and only report can comment."""
    jobs = job_blocks((WORKFLOWS / "stage.yml").read_text())
    for job, block in jobs.items():
        write_uses = re.findall(r"secrets\.HF_STAGING_TOKEN\b(?! != '')", block)
        read_uses = re.findall(r"secrets\.HF_STAGING_READ_TOKEN\b", block)
        assert not write_uses or job == "upload", job
        assert not read_uses or job == "download", job
        assert "secrets.HF_TOKEN" not in block and "secrets.HF_READ_TOKEN" not in block
        assert ("statuses: write" in block) == (job in ("target", "report")), job
        assert ("pull-requests: write" in block) == (job == "report"), job
    upload = jobs["upload"]
    name = re.search(r"^    environment:\n      name: (\S+)$", upload, re.M)
    assert name and name.group(1) == "hf-staging"
    assert "/environments/hf-staging" in jobs["target"]
    assert "--environment hf-staging --token-name HF_STAGING_TOKEN" in jobs["target"]
    assert 'find "pr/$KERNEL" -type l' in upload  # no symlinks into the upload
    assert "id-token" not in upload  # staging isn't signed


def test_kernel_builder_is_pinned():
    """The CI runs kernel-builder from the repository's flake.lock, never from the
    kernels default branch (the upload runs it with the Hub write token)."""
    files = sorted(WORKFLOWS.glob("*.yml")) + sorted(
        (REPO_ROOT / "scripts").glob("*.py")
    )
    for f in files:
        assert "github:huggingface/kernels#" not in f.read_text(), f.name


def test_every_workflow_upload_is_private():
    """kernel-builder creates public repositories unless told otherwise."""
    uploads = []
    for wf in sorted(WORKFLOWS.glob("*.yml")):
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


# The GPU runner comes from [ci-runners] in ci/community.toml, through the matrix.
GPU_RUNNERS = ("${{ inputs.runner }}", "${{ matrix.runner }}")


def test_runners_are_github_hosted_or_from_the_ci_table():
    """No workflow names a runner label: GPU jobs get theirs from [ci-runners], and
    everything else runs on GitHub-hosted Ubuntu."""
    for wf in sorted(WORKFLOWS.glob("*.yml")):
        for job, block in job_blocks(wf.read_text()).items():
            for runner in re.findall(r"^    runs-on: (.*)$", block, re.M):
                assert runner in GPU_RUNNERS or re.fullmatch(
                    r"ubuntu-[0-9.]+", runner
                ), f"{wf.name}: {job}: runs-on {runner}"


def test_gpu_jobs_run_in_the_ci_image():
    """Jobs on the GPU runners start a CI image themselves (`docker run`, not
    `container:`, so actions run on the runner with its own Node), with the GPU
    devices passed in, remove it at the end, and get no secrets: pull request code
    runs on those runners."""
    gpu_jobs = 0
    for wf in sorted(WORKFLOWS.glob("*.yml")):
        for job, block in job_blocks(wf.read_text()).items():
            runners = re.findall(r"^    runs-on: (.*)$", block, re.M)
            if not set(runners) & set(GPU_RUNNERS):
                continue
            gpu_jobs += 1
            where = f"{wf.name}: {job}"
            assert "secrets." not in block, f"{where}: uses a secret"
            assert not re.search(
                r"^    container:", block, re.M
            ), f"{where}: uses container:"
            image = (
                r"^          IMAGE: (ghcr\.io/rocm/rocm-community-kernels-ci:latest"
                r"|\$\{\{ vars\.CI_USER_IMAGE )"
            )
            assert re.search(image, block, re.M), f"{where}: not a known image"
            # Join shell continuation lines, so the command is one line.
            run = re.search(r"docker run .*", re.sub(r"\\\n\s*", " ", block))
            assert run, f"{where}: no docker run"
            assert '"$IMAGE"' in run.group(0), f"{where}: docker run not of $IMAGE"
            assert "--device /dev/kfd" in run.group(0), f"{where}: no /dev/kfd"
            assert "/etc/podinfo/gha-render-devices" in block, f"{where}: no podinfo"
            assert 'DEVICE_FLAG="--device /dev/dri"' in block, f"{where}: no /dev/dri"
            cleanup = re.search(
                r"^        if: always\(\)\n        run: \|\n(?:          .*\n|\n)*?"
                r"\s+docker rm -f ",
                block,
                re.M,
            )
            assert cleanup, f"{where}: no always() step removing the container"
    assert gpu_jobs >= 2  # rocm-build-test.yml build-test, publish.yml verify
