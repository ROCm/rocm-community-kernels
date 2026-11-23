"""List the kernels a change touches, for the pull request CI matrix.

A kernel is a top-level directory with a flake.nix and a build.toml (or
src/build.toml, for generated kernel trees). Deleted kernels
drop out because their directory no longer exists in the checkout.

Prints a JSON list of kernel directory names (--gpu-jobs: the GPU jobs instead).
With --github-output it also writes to $GITHUB_OUTPUT:

  kernels   the kernel directory names
  gpu-jobs  one {kernel, arch, runner} per GPU in a kernel's rocm-archs that CI has
            a runner for ([ci-runners] in ci/community.toml): the GPU test matrix

Directory names outside [A-Za-z0-9_-], and runner labels or GPU names outside
[A-Za-z0-9._-], are rejected: they end up in workflow commands.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_wrapping  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
SAFE_LABEL = re.compile(r"^[A-Za-z0-9._-]+$")


def is_kernel(root: Path, name: str) -> bool:
    k = root / name
    return (k / "flake.nix").is_file() and (
        (k / "build.toml").is_file() or (k / "src" / "build.toml").is_file()
    )


def all_kernels(root: Path) -> list[str]:
    return sorted(
        p.name for p in root.iterdir() if p.is_dir() and is_kernel(root, p.name)
    )


def kernels_in(paths: list[str], root: Path) -> list[str]:
    tops = {Path(p).parts[0] for p in paths if len(Path(p).parts) > 1}
    return sorted(t for t in tops if is_kernel(root, t))


def gpu_jobs(
    root: Path, kernels: list[str], runners: dict[str, str] | None = None
) -> list[dict]:
    """One job per GPU in each kernel's rocm-archs that CI has a runner for.

    A kernel whose build.toml doesn't parse gets no jobs; validate reports it.
    """
    if runners is None:
        runners = check_wrapping.load_community_rules()["ci-runners"]
    unsafe = [
        str(v) for kv in runners.items() for v in kv if not SAFE_LABEL.match(str(v))
    ]
    if unsafe:
        raise ValueError(
            f"[ci-runners] entries must match {SAFE_LABEL.pattern}: {unsafe}"
        )
    jobs = []
    for k in kernels:
        try:
            _, cfg = check_wrapping.load_config(root / k)
        except (OSError, tomllib.TOMLDecodeError):
            continue
        jobs += [
            {"kernel": k, "arch": arch, "runner": runners[arch]}
            for arch in check_wrapping.rocm_archs(cfg)
            if arch in runners
        ]
    return jobs


def changed_files(base: str, head: str, root: Path) -> list[str]:
    r = subprocess.run(
        ["git", "-C", str(root), "diff", "--name-only", f"{base}...{head}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in r.stdout.splitlines() if line]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    ap.add_argument("--base", help="base commit (the PR's base SHA)")
    ap.add_argument("--head", default="HEAD", help="head commit (default: HEAD)")
    ap.add_argument(
        "--kernels",
        help="comma-separated kernel names instead of a diff ('all' for every kernel)",
    )
    ap.add_argument("--gpu-jobs", action="store_true", help="print the GPU jobs")
    ap.add_argument("--github-output", action="store_true")
    args = ap.parse_args(argv)

    if args.kernels == "all":
        kernels = all_kernels(args.root)
    elif args.kernels:
        kernels = sorted({k.strip() for k in args.kernels.split(",") if k.strip()})
        missing = [
            k for k in kernels if SAFE_NAME.match(k) and not is_kernel(args.root, k)
        ]
        if missing:
            print(f"not kernel directories: {missing}", file=sys.stderr)
            return 1
    elif args.base:
        kernels = kernels_in(changed_files(args.base, args.head, args.root), args.root)
    else:
        ap.error("give --base or --kernels")

    unsafe = [k for k in kernels if not SAFE_NAME.match(k)]
    if unsafe:
        print(f"kernel names must match {SAFE_NAME.pattern}: {unsafe}", file=sys.stderr)
        return 1
    try:
        jobs = json.dumps(gpu_jobs(args.root, kernels))
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    out = json.dumps(kernels)
    print(jobs if args.gpu_jobs else out)
    if args.github_output:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"kernels={out}\ngpu-jobs={jobs}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
