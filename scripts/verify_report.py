"""Turn ci_verify_load.py's JSON reports into the text the workflows show.

  summary <verify.json>   the job summary of a verify job (publish.yml, stage.yml)
  stage ...               stage.yml's result: the commit status ("Hub staging
                          (<kernel>)") and the pull request comment, from every GPU's
                          report (verify-<arch>.json) and the result of each job

A verify job runs the kernel's tests, which come from the pull request, so its report
is only as trustworthy as Kernel CI's results: this reads known fields only, and puts
each value in an escaped, shortened code span, so a report can't add markup, links or
mentions to the comment. Standard library only: `summary` runs in the user image.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

FIELDS = ("variant", "commit", "device", "torch", "hip", "kernels", "tests_exit")
JOBS = ("target", "build", "upload", "download", "verify")
MAX_VALUE = 120


def status_context(kernel: str) -> str:
    """The commit status's name: one per kernel, so staging two kernels of one pull
    request doesn't overwrite one result with the other."""
    return f"Hub staging ({kernel})"


def code(value: object, limit: int = MAX_VALUE) -> str:
    """One value as an inline code span: no backticks, newlines or table pipes."""
    s = re.sub(r"[\x00-\x1f\x7f]+", " ", str(value)).replace("`", "'")
    if len(s) > limit:
        s = s[: limit - 1] + "…"
    return "`" + s.replace("|", "\\|") + "`"


def summary(report: dict) -> str:
    ok = "passed" if report.get("ok") else "FAILED"
    lines = [f"### get_kernel({code(report.get('repo_id'))}) from the Hub: {ok}", ""]
    for k in FIELDS + ("error",):
        if report.get(k) is not None:
            lines.append(f"- **{k}**: {code(report[k])}")
    return "\n".join(lines) + "\n"


def load_reports(directory: Path, archs: list[str]) -> dict:
    """Each GPU's report, verify-<arch>.json, anywhere below the downloaded artifacts
    (None if absent). download-artifact puts each artifact in a directory of its own
    only when it downloads more than one; with one GPU the report lands directly in
    `directory`, so the file name, not the directory, says which GPU it is."""
    out = {}
    for arch in archs:
        found = sorted(directory.rglob(f"verify-{arch}.json"))
        try:
            r = json.loads(found[0].read_text()) if found else None
            out[arch] = r if isinstance(r, dict) else None
        except (OSError, ValueError):
            out[arch] = None
    return out


def stage(
    kernel: str,
    sha: str,
    run_url: str,
    repo_id: str,
    branch: str,
    jobs: dict,
    reports: dict,
) -> tuple[str, str, str]:
    """(state, description, comment body) for stage.yml's result."""
    failed = [j for j in JOBS if jobs.get(j) != "success"]
    bad_gpus = [a for a, r in reports.items() if not (r and r.get("ok") is True)]
    ok = not failed and not bad_gpus and bool(reports)
    gpus = ", ".join(reports) or "no GPU"
    if ok:
        state, description = (
            "success",
            f"get_kernel() and kernels_ci tests passed: {gpus}",
        )
    else:
        state = "failure"
        # A GPU whose report failed also fails its verify job, so when verify is the
        # only failed job, name the GPUs instead.
        if bad_gpus and failed in ([], ["verify"]):
            description = f"get_kernel() or tests failed: {', '.join(bad_gpus)}"
        elif failed:
            description = f"failed at {failed[0]} ({jobs.get(failed[0]) or 'not run'})"
        else:
            description = "no verify report"

    lines = [
        f"<!-- hub-staging:{kernel} -->",
        f"### Hub staging: {code(kernel)} at {code(sha[:12])}: "
        + ("passed" if ok else "**failed**"),
        "",
        f"Uploaded privately to {code(repo_id)} (branch {code(branch)}), downloaded, "
        "loaded with `get_kernel()` in the user image, then the kernel's `kernels_ci` "
        "tests.",
        "",
    ]
    if reports:
        lines += [
            "| GPU | result | variant | torch | HIP | kernels | tests exit |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for arch, r in reports.items():
            if r is None:
                lines.append(f"| {code(arch)} | no report | | | | | |")
                continue
            cells = [r.get(k) for k in ("variant", "torch", "hip", "kernels")]
            cells.append(r.get("tests_exit"))
            row = " | ".join("" if c is None else code(c) for c in cells)
            res = "passed" if r.get("ok") is True else "failed"
            lines.append(f"| {code(arch)} | {res} | {row} |")
        errors = [(a, r["error"]) for a, r in reports.items() if r and r.get("error")]
        warnings = [
            (a, w)
            for a, r in reports.items()
            if r
            for w in (r.get("warnings") or [])[:3]
        ]
        if errors:
            lines += ["", "Errors:"] + [
                f"- {code(a)}: {code(e, 300)}" for a, e in errors
            ]
        if warnings:
            lines += ["", "Loader warnings:"]
            lines += [f"- {code(a)}: {code(w, 200)}" for a, w in warnings]
    lines += [
        "",
        "Jobs: " + ", ".join(f"{j} {code(jobs.get(j) or 'not run')}" for j in JOBS),
        "",
        f"[Run]({run_url}). These jobs run the pull request's code, as Kernel CI does. "
        "Staging is checked by an AMD maintainer and isn't required to merge; after "
        "merge, an AMD maintainer publishes the kernel with Publish kernel.",
    ]
    return state, description[:140], "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary", help="a verify job's summary, to stdout")
    s.add_argument("report", type=Path)
    st = sub.add_parser("stage", help="stage.yml's commit status and comment")
    st.add_argument("--kernel", required=True)
    st.add_argument("--sha", required=True)
    st.add_argument("--run-url", required=True)
    st.add_argument("--repo-id", required=True)
    st.add_argument("--branch", required=True)
    st.add_argument("--gpu-jobs", default="[]", help="target's gpu-jobs (JSON)")
    st.add_argument(
        "--jobs",
        required=True,
        help="each job's result: the workflow's toJSON(needs), or {job: result}",
    )
    st.add_argument(
        "--reports", type=Path, required=True, help="where the verify-* artifacts are"
    )
    st.add_argument("--status-json", type=Path, required=True)
    st.add_argument("--body", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.cmd == "summary":
        try:
            report = json.loads(args.report.read_text())
        except (OSError, ValueError) as e:
            print(f"### No verification report ({type(e).__name__})")
            return 0
        print(summary(report if isinstance(report, dict) else {}), end="")
        return 0

    try:
        archs = [j["arch"] for j in json.loads(args.gpu_jobs or "[]")]
    except (ValueError, KeyError, TypeError):
        archs = []
    reports = load_reports(args.reports, archs)
    jobs = {
        k: v.get("result") if isinstance(v, dict) else v
        for k, v in json.loads(args.jobs).items()
    }
    state, description, body = stage(
        args.kernel, args.sha, args.run_url, args.repo_id, args.branch, jobs, reports
    )
    context = status_context(args.kernel)
    args.status_json.write_text(
        json.dumps({"state": state, "description": description, "context": context})
        + "\n"
    )
    args.body.write_text(body)
    print(f"{context}: {state}: {description}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
