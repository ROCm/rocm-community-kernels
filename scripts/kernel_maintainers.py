"""Check the kernel maintainers registry, .github/kernel-maintainers.json.

The registry lists every kernel directory. The entries are GitHub handles
(without @), and every kernel names at least one maintainer (docs/COMMUNITY_POLICY.md).

Kernel maintainers are not CODEOWNERS: GitHub only accepts code owners with
write access to the repository, which outside contributors don't have. The
AMD maintainers team owns everything, and reviewers involve the kernel's
maintainers from this registry.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from changed_kernels import REPO_ROOT, all_kernels  # noqa: E402

REGISTRY = Path(".github/kernel-maintainers.json")
HANDLE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")


def check(root: Path) -> list[str]:
    registry = json.loads((root / REGISTRY).read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        return [
            f"{REGISTRY}: must be an object of kernel name -> list of GitHub handles"
        ]
    kernels = set(all_kernels(root))
    problems = [
        f"{k}: missing from {REGISTRY}" for k in sorted(kernels - set(registry))
    ]
    problems += [
        f"{k}: in {REGISTRY} but there is no kernel directory"
        for k in sorted(set(registry) - kernels)
    ]
    for k, handles in sorted(registry.items()):
        if not isinstance(handles, list) or not handles:
            problems.append(f"{k}: needs a list of at least one GitHub handle")
        elif bad := [
            h for h in handles if not isinstance(h, str) or not HANDLE.match(h)
        ]:
            problems.append(f"{k}: not GitHub handles (write them without @): {bad}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=["check"])
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    args = ap.parse_args(argv)
    problems = check(args.root)
    for p in problems:
        print(p, file=sys.stderr)
    if problems:
        return 1
    print(f"All {len(all_kernels(args.root))} kernel(s) name their maintainers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
