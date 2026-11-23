#!/usr/bin/env python3
"""Run the pull request CI stages for kernels.

  validate   check it is wrapped correctly for Hugging Face's kernel-builder
             (scripts/check_wrapping.py, with the community rules unless
             --hf-rules-only), then kernel-builder check-config
  build      nix build .#backendBundle.<backend> (every variant; --ci for one
             representative variant), then kernel-builder check-abi
  test       nix build the ci-test of the newest Torch with ROCm 7.1
             (.#ciTests.<variant>; --ci-test-rocm) and run it on the AMD GPU
             (pytest -m kernels_ci)

The GitHub workflows run this script (kernel-ci.yml, rocm-build-test.yml),
and contributors can run it the same way on a machine with Nix and an AMD GPU:

  python3 scripts/kernel_ci.py run --kernel <kernel> --ci

Writes <out>/<run-id>/summary.{json,md}, and per kernel findings.json, logs and
the build (<out>/<run-id>/<kernel>/build). Commands never get a Hugging Face
token: on the GPU runner they run pull request code.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

import check_wrapping  # noqa: E402

STAGES = ["validate", "build", "test"]
# The kernel-builder CLI pinned by the repository's flake.lock, not whatever is on the
# kernels default branch. (Builds use each kernel's own flake.lock.)
KERNEL_BUILDER = ["nix", "run", "-L", f"{REPO_ROOT}#kernel-builder", "--"]
SECRETS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_TOKEN")
BUILD_CONCURRENCY = REPO_ROOT / "build-concurrency.json"
# Hugging Face's binary cache holds the PyTorch x ROCm toolchains kernel-builder uses;
# without it Nix builds them from source. The workflows add it with cachix-action
# (kernel-ci.yml); a local machine or the GPU runner needs it in its Nix config.
HF_CACHE = "huggingface.cachix.org"
HF_CACHE_HINT = (
    "run `nix run nixpkgs#cachix -- use huggingface`, "
    "or install Nix with Hugging Face's kernels install.sh"
)
# Derivations that mean a toolchain is built from source: a cache miss costing hours.
TOOLCHAIN = re.compile(r"llvm|clang|rocm-clr|(?:^|-)torch-\d")
# The ROCm version the GPU test is built against. Kernel-builder's own `.#ci-test` is
# the newest Torch with its lowest ROCm (ROCm 7.14 at torch-2.15.0-final), whose
# toolchain isn't in the binary cache; the ROCm 7.1 one is.
CI_TEST_ROCM = "7.1"


@dataclass
class StageResult:
    status: str  # pass | fail | skip | dry-run
    seconds: float = 0.0
    message: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class KernelRun:
    entry: dict
    kernel_dir: Path
    out: Path
    info: check_wrapping.KernelInfo | None = None
    results: dict[str, StageResult] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.kernel_dir.name


def env_for(offline: bool, **extra: str) -> dict:
    """The environment for every command: never a Hugging Face token."""
    env = {k: v for k, v in os.environ.items() if k not in SECRETS}
    env.pop("LOCAL_KERNELS", None)
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
    else:
        env.pop("HF_HUB_OFFLINE", None)
    env.update(extra)
    return env


class Runner:
    def __init__(self, dry_run: bool):
        self.dry_run = dry_run

    def __call__(
        self, cmd: list, log: Path, cwd: Path | None = None, env: dict | None = None
    ) -> tuple[int, float]:
        cmd = [str(c) for c in cmd]
        line = (
            (f"(cd {shlex.quote(str(cwd))} && " if cwd else "")
            + shlex.join(cmd)
            + (")" if cwd else "")
        )
        print(f"    $ {line}", flush=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        if self.dry_run:
            with log.open("a") as f:
                f.write(f"[dry-run] {line}\n")
            return 0, 0.0
        t0 = time.monotonic()
        with log.open("a") as f:
            f.write(f"$ {line}\n")
            f.flush()
            rc = subprocess.run(
                cmd,
                cwd=cwd,
                env=env or env_for(True),
                stdout=f,
                stderr=subprocess.STDOUT,
            ).returncode
        return rc, time.monotonic() - t0


def detect() -> dict:
    kfd = Path("/dev/kfd")
    nix = shutil.which("nix") is not None
    return {
        "nix": nix,
        "hf_cache": nix_uses_hf_cache() if nix else None,
        "gpu": kfd.exists() and os.access(kfd, os.R_OK | os.W_OK),
        "rocminfo": shutil.which("rocminfo") is not None,
        "host": os.uname().nodename,
    }


def nix_uses_hf_cache() -> bool | None:
    """Whether Nix fetches from Hugging Face's binary cache (None: can't tell)."""
    for cmd in (["nix", "config", "show", "--json"], ["nix", "show-config", "--json"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0:
            try:
                return has_hf_cache(json.loads(r.stdout))
            except (ValueError, AttributeError, TypeError):
                return None
    return None


def has_hf_cache(config: dict) -> bool:
    """`nix config show --json` lists the cache as a substituter, and trusts its key."""
    substituters = config.get("substituters", {}).get("value", [])
    keys = config.get("trusted-public-keys", {}).get("value", [])
    return any(HF_CACHE in s for s in substituters) and any(
        k.startswith(f"{HF_CACHE}-") for k in keys
    )


def cache_check(ctx) -> StageResult | None:
    """A failure if Nix would build without Hugging Face's binary cache."""
    if ctx.dry_run or getattr(ctx.args, "no_hf_cache_check", False):
        return None
    if ctx.env.get("hf_cache") is False:
        return StageResult(
            "fail",
            message=f"Nix doesn't use Hugging Face's binary cache ({HF_CACHE}): "
            f"{HF_CACHE_HINT}; --no-hf-cache-check to build anyway",
        )
    return None


def source_builds(output: str) -> list[str]:
    """The derivations `nix build --dry-run` says it will build (not fetch), by name."""
    names, inside = [], False
    for ln in output.splitlines():
        if not ln.startswith(" "):
            inside = "will be built" in ln
        elif inside and ln.strip():
            names.append(re.sub(r"^/nix/store/[a-z0-9]+-|\.drv$", "", ln.strip()))
    return names


def toolchain_builds(ctx, attr: str, kr: KernelRun, log: Path) -> list[str]:
    """Toolchain derivations that building `attr` would compile from source."""
    if ctx.dry_run:
        return []
    cmd = ["nix", "build", "--dry-run", attr]
    r = subprocess.run(
        cmd, cwd=kr.kernel_dir, env=env_for(True), capture_output=True, text=True
    )
    with log.open("a") as f:
        f.write(f"$ (cd {shlex.quote(str(kr.kernel_dir))} && {shlex.join(cmd)})\n")
        f.write(r.stderr + r.stdout)
    heavy = [n for n in source_builds(r.stderr + r.stdout) if TOOLCHAIN.search(n)]
    if heavy:
        print(
            f"    WARNING {attr}: {len(heavy)} toolchain derivation(s) not in the "
            f"binary cache, built from source (hours): {', '.join(heavy[:5])}"
            + (", ..." if len(heavy) > 5 else ""),
            flush=True,
        )
    return heavy


def pick_ci_test(variants: list[str], rocm: str) -> str | None:
    """The variant with the newest Torch built against ROCm `rocm` (e.g. "7.1")."""
    tag = f"-rocm{rocm.replace('.', '')}-"

    def torch(v: str) -> tuple[int, int]:
        m = re.match(r"torch(\d)(\d+)-", v)
        return (int(m[1]), int(m[2]))

    found = [v for v in variants if tag in v and re.match(r"torch\d\d+-", v)]
    return max(found, key=torch, default=None)


def ci_test_attr(ctx, kr: KernelRun, log: Path) -> str:
    """The ci-test to build: `.#ciTests.<variant>` for --ci-test-rocm, else `.#ci-test`."""
    rocm = ctx.args.ci_test_rocm
    if not rocm:
        return ".#ci-test"
    if ctx.dry_run:
        return f".#ciTests.<newest Torch with ROCm {rocm}>"
    cmd = ["nix", "eval", "--json", ".#ciTests", "--apply", "builtins.attrNames"]
    r = subprocess.run(
        cmd, cwd=kr.kernel_dir, env=env_for(True), capture_output=True, text=True
    )
    with log.open("a") as f:
        f.write(f"$ (cd {shlex.quote(str(kr.kernel_dir))} && {shlex.join(cmd)})\n")
        f.write(r.stderr + r.stdout)
    try:
        variant = (
            pick_ci_test(json.loads(r.stdout), rocm) if r.returncode == 0 else None
        )
    except ValueError:
        variant = None
    if variant:
        return f".#ciTests.{variant}"
    print(
        f"    WARNING no ci-test variant with ROCm {rocm}; "
        "using kernel-builder's default .#ci-test",
        flush=True,
    )
    return ".#ci-test"


def git_sha(path: Path) -> str | None:
    r = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return r.stdout.strip() or None


KFD_NODES = Path("/sys/class/kfd/kfd/topology/nodes")


def kfd_gfx_names(nodes: Path = KFD_NODES) -> list[str]:
    """GPU names from the KFD topology, for when rocminfo isn't installed (the CI
    image has no ROCm of its own). gfx_target_version 90500 is gfx950, 90010 gfx90a."""
    names = set()
    for props in nodes.glob("*/properties"):
        try:
            text = props.read_text()
        except OSError:
            continue
        for ln in text.splitlines():
            key, _, value = ln.partition(" ")
            if key == "gfx_target_version" and value.strip().isdigit():
                v = int(value)
                if v:  # 0: a CPU node
                    names.add(f"gfx{v // 10000}{v // 100 % 100:x}{v % 100:x}")
    return sorted(names)


def gpu_info() -> dict:
    if not shutil.which("rocminfo"):
        names = kfd_gfx_names()
        return {"gfx": names} if names else {}
    r = subprocess.run(["rocminfo"], capture_output=True, text=True)
    names = sorted(
        {
            ln.split(":", 1)[1].strip()
            for ln in r.stdout.splitlines()
            if ln.strip().startswith("Name:") and "gfx" in ln
        }
    )
    return {"gfx": names}


def arch_mismatch(ctx, kr: KernelRun) -> str | None:
    """Why the GPU stages can't run here: this GPU isn't one the kernel targets, or
    not the one this run was routed to (--gpu-arch)."""
    if ctx.dry_run or not ctx.gpu_archs:
        return None
    wanted = getattr(getattr(ctx, "args", None), "gpu_arch", None)
    if wanted and wanted not in ctx.gpu_archs:
        return f"this run is for {wanted}; this GPU is {', '.join(ctx.gpu_archs)}"
    if not kr.info or not kr.info.rocm_archs:
        return None
    if set(kr.info.rocm_archs) & set(ctx.gpu_archs):
        return None
    return (
        f"kernel targets {', '.join(kr.info.rocm_archs)}; "
        f"this GPU is {', '.join(ctx.gpu_archs)}"
    )


def build_variants(build: Path) -> list[str]:
    """Variant folders in a kernel-builder output directory (each has metadata.json)."""
    return sorted(p.name for p in build.iterdir() if (p / "metadata.json").is_file())


# --------------------------------------------------------------------------- stages


def stage_validate(ctx, kr: KernelRun) -> StageResult:
    info, findings = check_wrapping.check_kernel(
        kr.kernel_dir,
        community=kr.entry.get("rules", "hf" if ctx.args.hf_rules_only else "community")
        == "community",
        require_backend=ctx.args.require_backend or None,
    )
    kr.info = info
    (kr.out / "findings.json").write_text(
        json.dumps([asdict(f) for f in findings], indent=2) + "\n"
    )
    errors = [f for f in findings if f.level == "error"]
    details = {
        "errors": len(errors),
        "warnings": len(findings) - len(errors),
        "findings": [asdict(f) for f in findings],
    }
    for f in findings:
        print(f"    {f.level.upper():7} {f.code}: {f.message}")
    if (ctx.env["nix"] or ctx.dry_run) and not ctx.args.no_check_config:
        rc, _ = ctx.run(
            [*KERNEL_BUILDER, "check-config", info.config_dir],
            kr.out / "validate.log",
            cwd=kr.kernel_dir,
        )
        details["kernel_builder_check_config"] = rc
        if rc:
            errors.append("kernel-builder check-config failed")
    if errors:
        return StageResult("fail", message=f"{len(errors)} error(s)", details=details)
    return StageResult(
        "pass", message=f"{details['warnings']} warning(s)", details=details
    )


def concurrency_args(kernel: str, backend: str) -> list[str]:
    """Nix job limits for memory-heavy kernels, from build-concurrency.json."""
    if not BUILD_CONCURRENCY.is_file():
        return []
    limits = json.loads(BUILD_CONCURRENCY.read_text()).get(kernel, {}).get(backend, {})
    return [
        arg
        for key in ("max-jobs", "cores")
        if key in limits
        for arg in (f"--{key}", str(limits[key]))
    ]


def stage_build(ctx, kr: KernelRun) -> StageResult:
    build = kr.out / "build"
    if not ctx.env["nix"] and not ctx.dry_run:
        return StageResult("skip", message="nix not found")
    if res := cache_check(ctx):
        return res
    backends = kr.entry.get("backends", ["rocm"])
    target = "backendCi" if ctx.args.ci else "backendBundle"
    log = kr.out / "build.log"
    heavy = []
    for b in backends:
        link = kr.out / f"result-{b}"
        heavy += toolchain_builds(ctx, f".#{target}.{b}", kr, log)
        rc, _ = ctx.run(
            [
                "nix",
                "build",
                "-L",
                f".#{target}.{b}",
                "--out-link",
                link,
                *concurrency_args(kr.name, b),
            ],
            log,
            cwd=kr.kernel_dir,
        )
        if rc:
            return StageResult(
                "fail", message=f"nix build .#{target}.{b} failed (see build.log)"
            )
        if not ctx.dry_run:
            shutil.copytree(link, build, symlinks=False, dirs_exist_ok=True)
    details = {"target": target, "backends": backends}
    if heavy:
        details["toolchain_from_source"] = heavy
    if kr.info and kr.info.framework != "torch-noarch":
        rc, _ = ctx.run(
            [*KERNEL_BUILDER, "check-abi", kr.kernel_dir], log, cwd=kr.kernel_dir
        )
        details["check_abi"] = rc
        if rc:
            return StageResult(
                "fail",
                message="kernel-builder check-abi failed (see build.log)",
                details=details,
            )
    if ctx.dry_run:
        return StageResult("dry-run", details=details)
    details["variants"] = build_variants(build)
    if not details["variants"]:
        return StageResult(
            "fail", message="the build has no variants (no <variant>/metadata.json)"
        )
    message = f"{len(details['variants'])} variant(s)"
    if heavy:
        message += f"; {len(heavy)} toolchain derivation(s) built from source"
    return StageResult("pass", message=message, details=details)


def stage_test(ctx, kr: KernelRun) -> StageResult:
    if why := arch_mismatch(ctx, kr):
        return StageResult("skip", message=why)
    if not ctx.env["nix"] and not ctx.dry_run:
        return StageResult("skip", message="nix not found")
    if not ctx.env["gpu"] and not ctx.args.force_gpu and not ctx.dry_run:
        return StageResult(
            "skip",
            message="no usable /dev/kfd (no AMD GPU, or no permission); --force-gpu to try anyway",
        )
    if res := cache_check(ctx):
        return res
    log = kr.out / "test.log"
    link = kr.out / "ci-test"
    attr = ci_test_attr(ctx, kr, log)
    heavy = toolchain_builds(ctx, attr, kr, log)
    rc, _ = ctx.run(
        ["nix", "build", "-L", attr, "--out-link", link], log, cwd=kr.kernel_dir
    )
    if rc:
        return StageResult("fail", message=f"nix build {attr} failed (see test.log)")
    rc, _ = ctx.run([link / "bin" / "ci-test"], log, cwd=kr.kernel_dir)
    status = "dry-run" if ctx.dry_run else ("pass" if rc == 0 else "fail")
    details = {"ci_test": attr, "gpu": gpu_info()}
    if heavy:
        details["toolchain_from_source"] = heavy
    return StageResult(
        status,
        message="" if rc == 0 else "ci-test failed (see test.log)",
        details=details,
    )


STAGE_FUNCS = {
    "validate": stage_validate,
    "build": stage_build,
    "test": stage_test,
}


# --------------------------------------------------------------------------- running


class Ctx:
    def __init__(self, args):
        self.args = args
        self.dry_run = args.dry_run
        self.run = Runner(args.dry_run)
        self.run_id = args.run_id or datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        self.out = (args.out / self.run_id).resolve()
        self.env = detect()
        self.gpu_archs = sorted({g.split(":")[0] for g in gpu_info().get("gfx", [])})


def select_stages(args, all_stages: list[str]) -> list[str]:
    stages = [
        s
        for s in (args.stages.split(",") if args.stages else all_stages)
        if s not in args.skip
    ]
    unknown = set(stages) - set(all_stages)
    if unknown:
        sys.exit(f"unknown stage(s): {sorted(unknown)}")
    return stages


def run_kernels(
    ctx, entries: list[dict], stages: list[str], stage_funcs: dict
) -> list[KernelRun]:
    """Run the stages for each kernel entry.

    An entry is {"path": ..., "_root": ...} plus optional keys: `rules` ("hf" or
    "community"), `backends`, `expect` ({stage: status}; a matching result doesn't
    fail the run), `keep_going` and `_label` (printed after the name).
    """
    ctx.out.mkdir(parents=True, exist_ok=True)
    print(f"run {ctx.run_id}: {len(entries)} kernel(s), stages {','.join(stages)}")
    print(f"  output {ctx.out}\n  detected {ctx.env}")
    runs = []
    for entry in entries:
        kdir = (entry["_root"] / entry["path"]).resolve()
        kr = KernelRun(entry, kdir, ctx.out / kdir.name)
        kr.out.mkdir(parents=True, exist_ok=True)
        runs.append(kr)
        print(f"\n== {kr.name}{entry.get('_label', '')}")
        if not kdir.is_dir():
            kr.results["validate"] = StageResult(
                "fail", message=f"no such directory {kdir}"
            )
            continue
        # Validation always runs: later stages need the parsed build.toml.
        order = ["validate"] + [s for s in stages if s != "validate"]
        failed = None
        for stage in order:
            if failed and not entry.get("keep_going", ctx.args.keep_going):
                kr.results[stage] = StageResult("skip", message=f"{failed} failed")
                continue
            if stage != "validate" and (
                kr.info is None or not kr.info.repo_id or kr.info.version is None
            ):
                kr.results[stage] = StageResult("skip", message="build.toml incomplete")
                continue
            print(f"  -- {stage}")
            t0 = time.monotonic()
            try:
                res = stage_funcs[stage](ctx, kr)
            except Exception as e:  # report and carry on with the next kernel
                res = StageResult("fail", message=f"{type(e).__name__}: {e}")
            if res.status == "skip" and ctx.args.fail_on_skip:
                # A required CI gate must not pass by not running.
                res = StageResult(
                    "fail", message=f"skipped, not allowed here: {res.message}"
                )
            res.seconds = round(time.monotonic() - t0, 1)
            kr.results[stage] = res
            expected = entry.get("expect", {}).get(stage)
            note = (
                ""
                if expected is None
                else (
                    " [as expected]"
                    if expected == res.status
                    else f" [expected {expected}]"
                )
            )
            print(f"     {res.status}{': ' + res.message if res.message else ''}{note}")
            if res.status == "fail":
                failed = stage
    return runs


def write_report(
    ctx,
    runs: list[KernelRun],
    stages: list[str],
    title: str = "Kernel CI",
    notes: list[str] = (),
    extra: dict | None = None,
) -> int:
    summary = {
        "run_id": ctx.run_id,
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "detected": ctx.env,
        **(extra or {}),
        "dry_run": ctx.dry_run,
        "kernels": [
            {
                "kernel": kr.name,
                "approved": kr.entry.get("approved", False),
                "repo_id": kr.info.repo_id if kr.info else None,
                "version": kr.info.version if kr.info else None,
                "stages": {s: asdict(r) for s, r in kr.results.items()},
                "expect": kr.entry.get("expect", {}),
                "unexpected": unexpected(kr),
            }
            for kr in runs
        ],
    }
    (ctx.out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    cols = ["validate"] + [s for s in stages if s != "validate"]
    lines = [f"# {title} {ctx.run_id}", ""]
    for n in notes:
        lines += [n, ""]
    header = len(lines)
    lines += [
        "| Kernel | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for kr in runs:
        cells = []
        for s in cols:
            r = kr.results.get(s)
            exp = kr.entry.get("expect", {}).get(s)
            mark = (
                ""
                if not r or exp is None
                else (" ✓ expected" if r.status == exp else f" ✗ expected {exp}")
            )
            cells.append(
                (f"{r.status}" + (f" ({r.message})" if r.message else "") + mark)
                if r
                else ""
            )
        lines.append(
            f"| {kr.name} | " + " | ".join(c.replace("|", "/") for c in cells) + " |"
        )
    (ctx.out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nsummary: {ctx.out / 'summary.md'}")
    print("\n".join(lines[header:]))
    bad = {kr.name: unexpected(kr) for kr in runs if unexpected(kr)}
    if bad:
        print(f"unexpected results: {bad}")
    return 1 if bad else 0


def unexpected(kr: KernelRun) -> list[str]:
    """Stages whose result isn't what the entry expects (by default: any failure)."""
    out = []
    for stage, res in kr.results.items():
        exp = kr.entry.get("expect", {}).get(stage)
        if exp is None and res.status == "fail":
            out.append(f"{stage}: fail")
        elif exp is not None and res.status not in (exp, "skip", "dry-run"):
            out.append(f"{stage}: {res.status}, expected {exp}")
    return out


def cmd_run(args) -> int:
    ctx = Ctx(args)
    stages = select_stages(args, STAGES)
    root = args.kernels_root or REPO_ROOT
    entries = [{"path": k, "_root": root} for k in args.kernel]
    if not entries:
        print("No kernels to run (use --kernel).")
        return 0
    runs = run_kernels(ctx, entries, stages, STAGE_FUNCS)
    return write_report(ctx, runs, stages)


# --------------------------------------------------------------------------- arguments


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "ci-out",
        help="output root (default: ci-out/)",
    )
    p.add_argument("--run-id", help="output sub-directory (default: a UTC timestamp)")
    p.add_argument(
        "--dry-run", action="store_true", help="print the commands without running them"
    )


def add_run_args(r: argparse.ArgumentParser, all_stages: list[str]) -> None:
    r.add_argument(
        "--kernels-root",
        type=Path,
        default=None,
        help="where kernel paths are resolved (default: the repository)",
    )
    r.add_argument(
        "--kernel",
        action="append",
        default=[],
        help="run this kernel (a directory under --kernels-root); repeatable",
    )
    r.add_argument("--stages", help=f"comma-separated subset of {','.join(all_stages)}")
    r.add_argument("--skip", action="append", default=[], help="skip a stage")
    r.add_argument(
        "--keep-going", action="store_true", help="run later stages after a failure"
    )
    r.add_argument(
        "--ci",
        action="store_true",
        help="build one representative variant (.#backendCi) instead of all",
    )
    r.add_argument(
        "--hf-rules-only",
        action="store_true",
        help="skip the rocm-community-kernels rules",
    )
    r.add_argument("--require-backend", default="rocm")
    r.add_argument(
        "--no-check-config",
        action="store_true",
        help="skip kernel-builder check-config",
    )
    r.add_argument(
        "--force-gpu",
        action="store_true",
        help="run the GPU test stage even if /dev/kfd looks unusable",
    )
    r.add_argument(
        "--ci-test-rocm",
        default=CI_TEST_ROCM,
        help="run the GPU test built for the newest Torch with this ROCm version "
        f"(default {CI_TEST_ROCM}, whose toolchain is in the binary cache); "
        "'' for kernel-builder's own .#ci-test",
    )
    r.add_argument(
        "--no-hf-cache-check",
        action="store_true",
        help=f"build even if Nix doesn't use Hugging Face's binary cache ({HF_CACHE})",
    )
    r.add_argument(
        "--gpu-arch",
        help="the GPU this run is for (e.g. gfx950): the test stage needs it present; "
        "the workflows pass the GPU each job was routed to",
    )
    r.add_argument(
        "--fail-on-skip",
        action="store_true",
        help="a stage that skips (no Nix, no GPU, GPU not targeted) fails instead; for CI gates",
    )


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the CI stages for the given kernels")
    add_common_args(r)
    add_run_args(r, STAGES)
    r.set_defaults(func=cmd_run)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
