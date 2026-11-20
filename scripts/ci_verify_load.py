"""Load a kernel with get_kernel() as a user would, then test or benchmark it.

Runs inside a user-like Python environment (PyTorch for ROCm plus the
`kernels` library, no Nix). The caller sets the environment:

* the Hub: KERNELS_CACHE for a clean download cache, and --trust-remote-code
  for organisations that aren't trusted publishers;
* an existing cache only: HF_HUB_OFFLINE=1 and KERNELS_CACHE=<cache directory>,
  for example one scripts/ci_download.py filled from a private repository. This
  script runs kernel code, so it never needs a token.

Checks that the module came from where it should (--expect-under), records the
build variant the loader chose and any loader warnings (digest, signature),
then optionally runs the kernel's `kernels_ci` tests, a smoke expression, or
the kernel's benchmarks/benchmark.py. Writes a JSON report; exit status 1 on
any failure.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import logging
import os
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(f"{record.levelname}: {record.getMessage()}")


def _variant_of(module_file: Path) -> tuple[str | None, str | None]:
    """(variant, commit) from .../snapshots/<commit>/build/<variant>/... ."""
    for p in [module_file, *module_file.parents]:
        if p.parent.name == "build":
            commit = (
                p.parent.parent.name
                if p.parent.parent.parent.name == "snapshots"
                else None
            )
            return p.name, commit
    return None, None


def _tensor_digest(t) -> dict:
    import torch

    t = t.detach().to("cpu").contiguous()
    try:
        raw = t.view(-1).view(torch.uint8).numpy().tobytes()
    except Exception:  # no numpy, or a dtype numpy can't view
        raw = repr(t.tolist()).encode()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _run_benchmarks(
    path: Path, kernel, device: str, iters: int, repeats: int
) -> list[dict]:
    """Time each benchmark_<workload> of the Benchmark subclasses in benchmarks/benchmark.py.

    Convention for benchmarks/benchmark.py:
    setup()/setup_<w>() prepares inputs, benchmark_<w>() runs the kernel and
    sets self.out, verify_<w>() returns the reference result.
    """
    import torch

    spec = importlib.util.spec_from_file_location("ci_benchmark_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        from kernels.benchmark import Benchmark
    except ImportError:
        Benchmark = None
    classes = [
        c
        for _, c in inspect.getmembers(mod, inspect.isclass)
        if c.__module__ == mod.__name__
        and (Benchmark is None or issubclass(c, Benchmark))
    ]
    gpu = device != "cpu" and torch.cuda.is_available()
    rows = []
    for cls in classes:
        try:
            inst = cls()
        except Exception as e:
            rows.append({"class": cls.__name__, "error": f"cannot instantiate: {e}"})
            continue
        inst.kernel, inst.device = kernel, device
        for name in sorted(n for n in dir(inst) if n.startswith("benchmark_")):
            w = name[len("benchmark_") :]
            row = {"class": cls.__name__, "workload": w}
            try:
                torch.manual_seed(getattr(inst, "seed", 0))
                setup = getattr(inst, f"setup_{w}", None) or getattr(
                    inst, "setup", None
                )
                if setup:
                    setup()
                run = getattr(inst, name)
                for _ in range(3):
                    run()
                verify = getattr(inst, f"verify_{w}", None)
                if verify is not None and hasattr(inst, "out"):
                    torch.testing.assert_close(inst.out, verify())
                    row["verified"] = True
                else:
                    row["verified"] = None
                times = []
                for _ in range(repeats):
                    if gpu:
                        torch.cuda.synchronize()
                        start, end = torch.cuda.Event(
                            enable_timing=True
                        ), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(iters):
                            run()
                        end.record()
                        torch.cuda.synchronize()
                        times.append(start.elapsed_time(end) / iters)
                    else:
                        t0 = time.perf_counter()
                        for _ in range(iters):
                            run()
                        times.append((time.perf_counter() - t0) * 1000 / iters)
                row.update(
                    median_ms=statistics.median(times),
                    min_ms=min(times),
                    iters=iters,
                    repeats=repeats,
                )
            except Exception as e:
                row.update(error=f"{type(e).__name__}: {e}", verified=False)
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repo-id", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--version", type=int)
    g.add_argument("--revision")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument(
        "--expect-under",
        type=Path,
        help="fail unless the module was loaded from below this directory",
    )
    ap.add_argument(
        "--tests", type=Path, help="run `pytest <dir> -m kernels_ci` after loading"
    )
    ap.add_argument(
        "--tests-repo-id",
        help="repo id the tests load (build.toml's), when it differs from --repo-id: "
        "points it at the build just loaded through LOCAL_KERNELS",
    )
    ap.add_argument(
        "--smoke",
        help="Python expression using k, torch and device; a tensor result is hashed",
    )
    ap.add_argument("--benchmark", type=Path, help="benchmarks/benchmark.py to time")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args(argv)

    report: dict = {
        "repo_id": args.repo_id,
        "version": args.version,
        "revision": args.revision,
        "offline": os.environ.get("HF_HUB_OFFLINE") == "1",
        "kernels_cache": os.environ.get("KERNELS_CACHE"),
        "python": platform.python_version(),
        "ok": False,
    }
    collect = _Collect()
    logging.getLogger("kernels").addHandler(collect)
    logging.captureWarnings(True)
    logging.getLogger("py.warnings").addHandler(collect)
    ok = True
    try:
        import torch
        import kernels

        report.update(
            torch=torch.__version__,
            hip=getattr(torch.version, "hip", None),
            kernels=getattr(kernels, "__version__", None),
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        report["device"] = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"

        kw = (
            {"version": args.version}
            if args.version is not None
            else {"revision": args.revision}
        )
        if args.trust_remote_code:
            kw["trust_remote_code"] = True
        k = kernels.get_kernel(args.repo_id, **kw)
        module_file = Path(
            os.path.abspath(k.__file__)
        )  # not resolve(): Hub cache files are symlinks
        variant, commit = _variant_of(module_file)
        report.update(module_file=str(module_file), variant=variant, commit=commit)
        variant_dir = next(
            (p for p in module_file.parents if p.parent.name == "build"), None
        )
        if variant_dir is not None:
            report["variant_dir"] = str(variant_dir)
            report["module_relpath"] = module_file.relative_to(variant_dir).as_posix()
        if args.expect_under is not None:
            under = Path(os.path.abspath(args.expect_under))
            report["expected_under"] = str(under)
            if under not in module_file.parents:
                ok = False
                report["error"] = f"loaded {module_file}, which is not under {under}"

        if ok and args.smoke:
            result = eval(
                args.smoke, {"k": k, "torch": torch, "device": device}
            )  # noqa: S307 (CI-supplied)
            report["smoke"] = (
                _tensor_digest(result)
                if isinstance(result, torch.Tensor)
                else repr(result)
            )

        if ok and args.tests is not None:
            import pytest

            if args.tests_repo_id and args.tests_repo_id != args.repo_id:
                # The tests call get_kernel(<build.toml repo id>); a publish to a test
                # organisation lands elsewhere, so map that id to what was just loaded.
                if variant_dir is None:
                    raise RuntimeError(
                        f"no build variant directory above {module_file}"
                    )
                os.environ["LOCAL_KERNELS"] = f"{args.tests_repo_id}={variant_dir}"
                report["tests_local_kernels"] = os.environ["LOCAL_KERNELS"]

            rc = pytest.main(
                [str(args.tests), "-m", "kernels_ci", "-p", "no:cacheprovider", "-q"]
            )
            # As in the Nix ci-test: a suite with no kernels_ci tests is not a failure.
            report["tests_exit"] = int(rc)
            ok = rc in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED)

        if ok and args.benchmark is not None:
            rows = _run_benchmarks(args.benchmark, k, device, args.iters, args.repeats)
            report["benchmarks"] = rows
            ok = bool(rows) and all(
                "error" not in r and r.get("verified") is not False for r in rows
            )
    except Exception as e:
        ok = False
        report["error"] = f"{type(e).__name__}: {e}"
        report["traceback"] = traceback.format_exc()

    report["warnings"] = collect.messages
    report["digest_warnings"] = [m for m in collect.messages if "digest" in m.lower()]
    report["ok"] = ok
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: report.get(k) for k in ("ok", "variant", "module_file", "error")},
            indent=2,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
