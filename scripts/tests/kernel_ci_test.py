import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kernel_ci  # noqa: E402
from check_wrapping import KernelInfo  # noqa: E402


def kernel(archs, expect=None, results=None):
    info = KernelInfo("k", "/k", "/k", "org/k", 1, "torch", ["rocm"], "k", archs)
    kr = kernel_ci.KernelRun(
        {"path": "k", "expect": expect or {}}, Path("/k"), Path("/out"), info
    )
    kr.results = results or {}
    return kr


def ctx(gpu_archs, dry_run=False):
    return SimpleNamespace(gpu_archs=gpu_archs, dry_run=dry_run)


def test_arch_gating():
    assert kernel_ci.arch_mismatch(ctx(["gfx950"]), kernel(["gfx1201"])) is not None
    assert (
        kernel_ci.arch_mismatch(ctx(["gfx950"]), kernel(["gfx942", "gfx950"])) is None
    )
    # no archs: may load anywhere
    assert kernel_ci.arch_mismatch(ctx(["gfx950"]), kernel([])) is None
    # GPU unknown: try
    assert kernel_ci.arch_mismatch(ctx([]), kernel(["gfx1201"])) is None
    assert (
        kernel_ci.arch_mismatch(ctx(["gfx950"], dry_run=True), kernel(["gfx1201"]))
        is None
    )


def test_gpu_arch_routing():
    """--gpu-arch: the job must be on the GPU it was routed to."""
    routed = SimpleNamespace(
        gpu_archs=["gfx950"], dry_run=False, args=SimpleNamespace(gpu_arch="gfx942")
    )
    why = kernel_ci.arch_mismatch(routed, kernel(["gfx942", "gfx950"]))
    assert why and "gfx942" in why
    routed.args.gpu_arch = "gfx950"
    assert kernel_ci.arch_mismatch(routed, kernel(["gfx942", "gfx950"])) is None


def test_kfd_gfx_names(tmp_path):
    """The GPU names from the KFD topology, when rocminfo isn't installed."""
    for node, version in (("0", 0), ("1", 90500), ("2", 90010), ("3", 110001)):
        (tmp_path / node).mkdir()
        (tmp_path / node / "properties").write_text(
            f"cpu_cores_count 0\ngfx_target_version {version}\nsimd_count 256\n"
        )
    assert kernel_ci.kfd_gfx_names(tmp_path) == ["gfx1101", "gfx90a", "gfx950"]
    assert kernel_ci.kfd_gfx_names(tmp_path / "missing") == []


NOARCH_KERNEL = {
    "flake.nix": "{}\n",
    "build.toml": (
        '[general]\nname = "demo"\nversion = 1\nedition = 6\nlicense = "MIT"\n'
        'backends = ["rocm"]\n\n[general.hub]\nrepo-id = "rocm-community/demo"\n\n'
        '[torch-noarch]\nrocm-archs = ["gfx950"]\n\n[kernel]\n'
    ),
    "README.md": "# demo\n\nThis kernel is not supported by AMD.\n",
    "LICENSE": "MIT\n",
    "torch-ext/demo/__init__.py": "def f():\n    pass\n",
    "tests/demo_test.py": (
        "import kernels\nimport pytest\n\n"
        'demo = kernels.get_kernel("rocm-community/demo", version=1)\n\n\n'
        "@pytest.mark.kernels_ci\ndef test_demo():\n    assert demo\n"
    ),
}
PARSE_ERROR = (
    "Error: Cannot parse build configuration: Unknown field rocm-archs. "
    "Expected one of pyext, cuda_capabilities, rocm_archs"
)


def test_validate_explains_its_errors(tmp_path, monkeypatch, capsys):
    """A mistake shows what to fix: the finding, check-config's own error, and in
    GitHub Actions an annotation on the file."""
    k = tmp_path / "demo"
    for rel, text in NOARCH_KERNEL.items():
        (k / rel).parent.mkdir(parents=True, exist_ok=True)
        (k / rel).write_text(text)

    def run(cmd, log, cwd=None):
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(f"$ check-config\ncopying path...\n{PARSE_ERROR}\n")
        return 1, 0.0

    args = SimpleNamespace(hf_rules_only=False, require_backend="rocm")
    args.no_check_config = False
    ctx = SimpleNamespace(args=args, env={"nix": True}, dry_run=False, run=run)
    kr = kernel_ci.KernelRun({"path": "demo"}, k, tmp_path / "out")
    kr.out.mkdir()
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    res = kernel_ci.stage_validate(ctx, kr)
    out = capsys.readouterr().out
    assert res.status == "fail"
    assert res.message == "2 error(s): noarch-archs-key, check-config"
    assert "write rocm_archs" in out  # the finding says what to write
    assert PARSE_ERROR in res.details["check_config_error"]
    assert "::error file=" in out and "title=demo%3A noarch-archs-key::" in out
    assert "title=demo%3A check-config::kernel-builder check-config failed" in out


def test_summary_lists_validate_errors(tmp_path, monkeypatch):
    (tmp_path / "k").mkdir()
    finding = {"level": "error", "code": "noarch-archs-key", "message": "write it"}
    details = {"findings": [finding], "check_config_error": "check-config failed: x"}
    monkeypatch.setitem(
        kernel_ci.STAGE_FUNCS,
        "validate",
        lambda ctx, kr: kernel_ci.StageResult("fail", message="2", details=details),
    )
    args = ["run", "--kernels-root", str(tmp_path), "--kernel", "k"]
    args += ["--stages", "validate", "--out", str(tmp_path / "out"), "--run-id", "r"]
    assert kernel_ci.main(args) == 1
    summary = (tmp_path / "out" / "r" / "summary.md").read_text()
    assert "- k: `noarch-archs-key`: write it" in summary
    assert "- k: check-config failed: x" in summary


def test_annotations(monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    kernel_ci.annotate("error", "t", "m")  # outside GitHub Actions: nothing
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    path = kernel_ci.REPO_ROOT / "demo" / "build.toml"
    kernel_ci.annotate("warning", "demo: x", "50% done\nnext", path)
    assert capsys.readouterr().out == (
        "::warning file=demo/build.toml,title=demo%3A x::50%25 done%0Anext\n"
    )


def test_fail_on_skip(tmp_path, monkeypatch):
    (tmp_path / "k").mkdir()

    def validate(ctx, kr):
        kr.info = kernel([]).info
        return kernel_ci.StageResult("pass")

    monkeypatch.setitem(kernel_ci.STAGE_FUNCS, "validate", validate)
    monkeypatch.setitem(
        kernel_ci.STAGE_FUNCS,
        "test",
        lambda ctx, kr: kernel_ci.StageResult("skip", message="no GPU"),
    )
    base = ["run", "--kernels-root", str(tmp_path), "--kernel", "k"]
    base += ["--stages", "validate,test", "--out", str(tmp_path / "out")]
    assert kernel_ci.main([*base, "--run-id", "lenient"]) == 0
    assert kernel_ci.main([*base, "--run-id", "strict", "--fail-on-skip"]) == 1
    summary = (tmp_path / "out" / "strict" / "summary.md").read_text()
    assert "skipped, not allowed here: no GPU" in summary


def test_concurrency_args(tmp_path, monkeypatch):
    limits = tmp_path / "build-concurrency.json"
    limits.write_text('{"big": {"rocm": {"max-jobs": 1, "cores": 8}}}')
    monkeypatch.setattr(kernel_ci, "BUILD_CONCURRENCY", limits)
    assert kernel_ci.concurrency_args("big", "rocm") == [
        "--max-jobs",
        "1",
        "--cores",
        "8",
    ]
    assert kernel_ci.concurrency_args("small", "rocm") == []


def test_hf_cache_detection():
    key = "huggingface.cachix.org-1:<public key>"
    nixos = ["https://cache.nixos.org/"]

    def config(substituters, keys):
        return {
            "substituters": {"value": substituters},
            "trusted-public-keys": {"value": keys},
        }

    assert kernel_ci.has_hf_cache(
        config([*nixos, "https://huggingface.cachix.org"], [key])
    )
    assert not kernel_ci.has_hf_cache(config(nixos, []))
    # the key alone doesn't make Nix fetch from it, nor the URL without a trusted key
    assert not kernel_ci.has_hf_cache(config(nixos, [key]))
    assert not kernel_ci.has_hf_cache(
        config(["https://huggingface.cachix.org"], ["cache.nixos.org-1:x"])
    )
    assert not kernel_ci.has_hf_cache({})


def test_missing_hf_cache_fails_before_building(tmp_path):
    def ctx(hf_cache, **args):
        return SimpleNamespace(
            dry_run=False,
            env={"nix": True, "hf_cache": hf_cache},
            args=SimpleNamespace(ci=True, no_hf_cache_check=False, **args),
        )

    res = kernel_ci.stage_build(ctx(False), kernel([]))
    assert res.status == "fail" and "cachix -- use huggingface" in res.message
    # Nix couldn't say, or the user opted out: don't block
    assert kernel_ci.cache_check(ctx(None)) is None
    opted_out = ctx(False)
    opted_out.args.no_hf_cache_check = True
    assert kernel_ci.cache_check(opted_out) is None


def test_source_builds_from_nix_dry_run():
    out = """\
these 3 derivations will be built:
  /nix/store/0a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p-rocm-llvm-7.14.0.drv
  /nix/store/1a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p-python3.13-torch-2.10.0.drv
  /nix/store/2a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p-sage-attention-torch-ext.drv
these 2 paths will be fetched (1.20 GiB download, 6.00 GiB unpacked):
  /nix/store/3a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p-rocm-clr-7.1.1
  /nix/store/4a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p-clang-19.1.0
"""
    names = kernel_ci.source_builds(out)
    assert names == [
        "rocm-llvm-7.14.0",
        "python3.13-torch-2.10.0",
        "sage-attention-torch-ext",
    ]
    # fetched paths and the kernel's own derivation aren't toolchain builds
    assert [n for n in names if kernel_ci.TOOLCHAIN.search(n)] == names[:2]
    assert (
        kernel_ci.source_builds("this path will be fetched:\n  /nix/store/x-a\n") == []
    )


def test_ci_test_uses_the_newest_torch_for_the_rocm_version():
    variants = [
        "torch29-rocm71-x86_64-linux",
        "torch210-rocm70-x86_64-linux",
        "torch210-rocm71-x86_64-linux",
        "torch213-rocm71-x86_64-linux",
        "torch214-rocm714-x86_64-linux",
        "torch215-rocm714-x86_64-linux",
        "torch215-rocm100-x86_64-linux",
    ]
    assert kernel_ci.pick_ci_test(variants, "7.1") == "torch213-rocm71-x86_64-linux"
    # rocm71 must not match rocm714
    assert kernel_ci.pick_ci_test(variants, "7.14") == "torch215-rocm714-x86_64-linux"
    assert kernel_ci.pick_ci_test(variants, "7.2") is None
    assert kernel_ci.pick_ci_test([], "7.1") is None


def test_expectations():
    R = kernel_ci.StageResult
    ok = kernel([], {"validate": "fail"}, {"validate": R("fail"), "build": R("pass")})
    assert kernel_ci.unexpected(ok) == []
    wrong = kernel([], {"validate": "fail"}, {"validate": R("pass")})
    assert kernel_ci.unexpected(wrong) == ["validate: pass, expected fail"]
    plain = kernel([], {}, {"build": R("fail"), "test": R("skip")})
    assert kernel_ci.unexpected(plain) == ["build: fail"]


def test_commands_never_get_a_token(monkeypatch):
    """Pull request code runs on the GPU runner: no command may see a Hub token."""
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "secret")
    monkeypatch.setenv("LOCAL_KERNELS", "x=y")
    env = kernel_ci.env_for(True, KERNELS_CACHE="/tmp/hub")
    assert (
        "HF_TOKEN" not in env
        and "HUGGING_FACE_HUB_TOKEN" not in env
        and "LOCAL_KERNELS" not in env
    )
    assert env["HF_HUB_OFFLINE"] == "1" and env["KERNELS_CACHE"] == "/tmp/hub"
    assert "HF_HUB_OFFLINE" not in kernel_ci.env_for(False)
