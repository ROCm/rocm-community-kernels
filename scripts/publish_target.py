"""Work out where the publish workflow uploads a kernel, and how to load it back.

The organisation comes from the workflow (its hub_org input, or the HUB_ORG
repository variable); the repository name and version come from build.toml.
The upload goes to branch v<version>, and version-1 kernels without an
explicit branch also go to main.

Writes these to $GITHUB_OUTPUT (or prints them):

  repo-id         where to upload, <hub-org>/<name>
  source-repo-id  build.toml's repo-id, which the kernel's tests load
  branch          the Hub branch to upload to
  also-main       "true" when main gets the same upload
  load-args       ci_verify_load.py arguments to load that branch as a user would
  config-dir      the directory holding build.toml (and tests/)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_wrapping  # noqa: E402
from changed_kernels import REPO_ROOT, SAFE_NAME  # noqa: E402

BRANCH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ORG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def target(root: Path, kernel: str, hub_org: str, branch: str = "") -> dict:
    if not SAFE_NAME.match(kernel):
        raise ValueError(f"kernel name must match {SAFE_NAME.pattern}: {kernel!r}")
    if not ORG.match(hub_org or ""):
        raise ValueError(
            f"invalid Hub organisation {hub_org!r}: set the hub_org input or the HUB_ORG variable"
        )
    rules = check_wrapping.load_community_rules()
    if hub_org.lower() in rules["reserved-orgs"]:
        raise ValueError(f"{hub_org!r} is reserved for AMD's own packages")
    if branch and not BRANCH.match(branch):
        raise ValueError(f"invalid branch name {branch!r}")
    config_dir, cfg = check_wrapping.load_config(root / kernel)
    general = cfg.get("general", {})
    source = general.get("hub", {}).get("repo-id")
    version = general.get("version")
    if not source or "/" not in source:
        raise ValueError(f"{kernel}: build.toml has no [general.hub] repo-id")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError(f"{kernel}: build.toml has no integer [general] version")
    out_branch = branch or f"v{version}"
    return {
        "repo-id": f"{hub_org}/{source.split('/', 1)[1]}",
        "source-repo-id": source,
        "branch": out_branch,
        "also-main": "true" if version == 1 and not branch else "false",
        "load-args": (
            f"--version {version}"
            if out_branch == f"v{version}"
            else f"--revision {out_branch}"
        ),
        "config-dir": str(config_dir.relative_to(root.resolve())),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("kernel")
    ap.add_argument("--hub-org", required=True)
    ap.add_argument("--branch", default="")
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    ap.add_argument("--github-output", action="store_true")
    args = ap.parse_args(argv)
    try:
        t = target(args.root.resolve(), args.kernel, args.hub_org, args.branch)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    lines = [f"{k}={v}" for k, v in t.items()]
    print("\n".join(lines))
    if args.github_output:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
