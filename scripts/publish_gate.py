"""Check that the publish workflow's approval gate is in place, before anything runs.

publish.yml's upload job holds the Hub write token (HF_TOKEN) and waits for approval
in the hf-rocm-community environment. That gate only exists if:

  - the environment exists: otherwise GitHub creates it, with no protection, on the
    first run that names it;
  - it has required reviewers;
  - it limits which branches can deploy to it;
  - HF_TOKEN is a secret of that environment. A repository or organisation secret of
    that name is visible to every job, so the upload would need no approval.

Reads the environment as `GET /repos/{repo}/environments/{name}` returns it (a missing
file means the request failed), and whether HF_TOKEN is visible to the calling job,
which isn't in the environment. Prints each problem; exit status 1 if there is one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ENVIRONMENT = "hf-rocm-community"
SETUP = "see Repository setup in ci/README.md"


def problems(environment: dict | None, write_token_visible: bool) -> list[str]:
    out = []
    if write_token_visible:
        out.append(
            "HF_TOKEN is visible outside the environment: it is a repository or "
            "organisation secret, so the upload would need no approval. Delete it "
            f"there and add it as a secret of the {ENVIRONMENT} environment"
        )
    if environment is None:
        out.append(
            f"the {ENVIRONMENT} environment doesn't exist (or this job can't read it); "
            "GitHub would create it with no required reviewers"
        )
        return out
    rules = environment.get("protection_rules") or []
    if not any(
        r.get("type") == "required_reviewers" and r.get("reviewers") for r in rules
    ):
        out.append(f"the {ENVIRONMENT} environment has no required reviewers")
    if not environment.get("deployment_branch_policy"):
        out.append(
            f"the {ENVIRONMENT} environment allows deployments from any branch; "
            "limit it to main"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--environment-json",
        type=Path,
        required=True,
        help="the environment as the GitHub API returns it (missing: not found)",
    )
    ap.add_argument(
        "--write-token-visible",
        choices=["true", "false"],
        required=True,
        help="whether HF_TOKEN is visible to this job, outside the environment",
    )
    args = ap.parse_args(argv)
    environment = None
    if args.environment_json.is_file():
        environment = json.loads(args.environment_json.read_text())
    found = problems(environment, args.write_token_visible == "true")
    for p in found:
        print(f"error: {p}", file=sys.stderr)
    if found:
        print(f"The publish approval gate isn't in place: {SETUP}", file=sys.stderr)
        return 1
    print(f"Approval gate in place: {ENVIRONMENT} has required reviewers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
