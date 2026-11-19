"""Download a kernel from the Hub into a cache, without loading it.

The first half of a split verify: this step may hold a read token (for private
repositories) because it only downloads files; nothing from the kernel is
imported or run. The second half, scripts/ci_verify_load.py, then loads the
kernel from that cache with HF_HUB_OFFLINE=1 and no token, and runs its tests.

The caller sets the environment:

* KERNELS_CACHE: the cache to download into, which ci_verify_load.py reads;
* HF_TOKEN: a read token, when the repository is private.

Downloads `build/` of the given branch with `huggingface_hub.snapshot_download`,
which records refs/<branch> in the cache, so get_kernel(version=N) resolves the
branch offline afterwards. Writes a JSON report (no token in it); exit status 1
on failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Byte-code isn't part of a kernel; the `kernels` loader skips it too.
IGNORE = ["*.pyc", "*/__pycache__/*"]


def download(repo_id: str, revision: str, cache_dir: str) -> dict:
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id,
            repo_type="kernel",
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns="build/*",
            ignore_patterns=IGNORE,
        )
    )
    build = snapshot / "build"
    variants = (
        sorted(p.name for p in build.iterdir() if (p / "metadata.json").is_file())
        if build.is_dir()
        else []
    )
    return {"snapshot": str(snapshot), "commit": snapshot.name, "variants": variants}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--revision", required=True, help="branch to download, e.g. v1")
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args(argv)

    report: dict = {"repo_id": args.repo_id, "revision": args.revision, "ok": False}
    cache = os.environ.get("KERNELS_CACHE")
    try:
        if not cache:
            raise RuntimeError("set KERNELS_CACHE to the cache to download into")
        report["kernels_cache"] = cache
        report.update(download(args.repo_id, args.revision, cache))
        if not report["variants"]:
            raise RuntimeError(f"no build variants in {args.repo_id}@{args.revision}")
        report["ok"] = True
    except Exception as e:
        report["error"] = f"{type(e).__name__}: {e}"

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: report.get(k) for k in ("ok", "commit", "variants", "error")}, indent=2
        )
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
