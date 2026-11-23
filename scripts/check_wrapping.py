"""Check that a kernel directory is wrapped correctly for Hugging Face's kernel-builder.

Checks the packaging rules in AGENTS.md: a flake and an edition-6 build.toml
with a Hub repo id, a Python package that only imports itself relatively, ops
registered under the build-generated namespace, and tests that load the kernel
through `get_kernel()` with a `kernels_ci` subset. `--community` adds the rules for
ROCm/rocm-community-kernels from ci/community.toml: the community Hub
organisation, no AITER code, no AMD-reserved names, and the card notice.

Exit status is 1 if any kernel has an error.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

COMMUNITY_CONFIG = Path(__file__).resolve().parents[1] / "ci" / "community.toml"
COMMUNITY_DEFAULTS = {
    "hub-org": "rocm-community",
    "reserved-prefixes": ["aiter"],
    "reserved-orgs": ["amd", "amd-aiter"],
    "ci-archs": ["gfx950"],
    "card-notice": "not supported by AMD",
}
SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
FRAMEWORKS = ("torch", "torch-noarch", "tvm-ffi")
# C++ registration with a fixed namespace (the *_EXPAND forms take TORCH_EXTENSION_NAME).
FIXED_CPP_NS = re.compile(r"\b(?:STABLE_)?TORCH_LIBRARY(?:_IMPL|_FRAGMENT)?\s*\(")
# Python registration with a literal "namespace::op" name.
FIXED_PY_NS = re.compile(
    r"\b(?:custom_op|register_fake|register_kernel|register_autograd|define|impl)\s*\(\s*[rbuf]*[\"'][\w.]+::"
)


@dataclass
class Finding:
    level: str  # "error" or "warning"
    code: str
    message: str
    path: str = ""


@dataclass
class KernelInfo:
    name: str
    kernel_dir: str
    config_dir: str
    repo_id: str | None
    version: int | None
    framework: str | None
    backends: list[str]
    package: str | None
    rocm_archs: list[str] = field(default_factory=list)  # GPU targets in build.toml


def find_config_dir(kernel_dir: Path) -> Path:
    """Generated kernel trees keep build.toml in <kernel>/src, with flake.nix in <kernel>."""
    return (
        kernel_dir / "src"
        if (kernel_dir / "src" / "build.toml").is_file()
        else kernel_dir
    )


def load_config(kernel_dir: Path) -> tuple[Path, dict]:
    config_dir = find_config_dir(kernel_dir)
    with open(config_dir / "build.toml", "rb") as f:
        return config_dir, tomllib.load(f)


def check_kernel(
    kernel_dir: Path, *, community: bool = False, require_backend: str | None = "rocm"
) -> tuple[KernelInfo, list[Finding]]:
    kernel_dir = kernel_dir.resolve()
    findings: list[Finding] = []

    def err(code, msg, path=""):
        findings.append(Finding("error", code, msg, str(path)))

    def warn(code, msg, path=""):
        findings.append(Finding("warning", code, msg, str(path)))

    info = KernelInfo(
        kernel_dir.name, str(kernel_dir), str(kernel_dir), None, None, None, [], None
    )
    if not SAFE_NAME.match(kernel_dir.name):
        err(
            "name", f"directory name {kernel_dir.name!r} must match {SAFE_NAME.pattern}"
        )
    if not (kernel_dir / "flake.nix").is_file():
        err("flake", "flake.nix is missing", kernel_dir)
    config_dir = find_config_dir(kernel_dir)
    info.config_dir = str(config_dir)
    toml_path = config_dir / "build.toml"
    if not toml_path.is_file():
        err("build-toml", "build.toml is missing", toml_path)
        return info, findings
    try:
        _, cfg = load_config(kernel_dir)
    except tomllib.TOMLDecodeError as e:
        err("build-toml", f"build.toml does not parse: {e}", toml_path)
        return info, findings

    general = cfg.get("general", {})
    name = general.get("name")
    if not name:
        err("general-name", "[general] name is missing", toml_path)
        name = kernel_dir.name
    elif name != kernel_dir.name:
        warn(
            "general-name",
            f"[general] name {name!r} differs from the directory name {kernel_dir.name!r}",
            toml_path,
        )
    info.name = name
    if general.get("edition") != 6:
        warn(
            "edition",
            "[general] edition should be 6 (older files are upgraded with a warning)",
            toml_path,
        )
    version = general.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        err(
            "version",
            "[general] version must be a non-negative integer (it selects the Hub branch vN)",
            toml_path,
        )
    else:
        info.version = version
    if not general.get("license"):
        err("license", "[general] license is missing", toml_path)
    repo_id = general.get("hub", {}).get("repo-id")
    if not repo_id:
        err(
            "repo-id",
            "[general.hub] repo-id is missing (it names the Hub repository to publish to)",
            toml_path,
        )
    elif not REPO_ID.match(repo_id):
        err("repo-id", f"repo-id {repo_id!r} must look like <org>/<name>", toml_path)
    info.repo_id = repo_id
    backends = general.get("backends", [])
    info.backends = list(backends)
    if require_backend and require_backend not in backends:
        err(
            "backends",
            f"[general] backends must include {require_backend!r}",
            toml_path,
        )

    frameworks = [f for f in FRAMEWORKS if f in cfg]
    if len(frameworks) != 1:
        err(
            "framework",
            f"build.toml needs exactly one of {FRAMEWORKS}, found {frameworks or 'none'}",
            toml_path,
        )
    framework = frameworks[0] if frameworks else None
    info.framework = framework
    archs = set(cfg.get("torch-noarch", {}).get("rocm-archs", []))
    for section in cfg.get("kernel", {}).values():
        if isinstance(section, dict):
            archs |= set(section.get("rocm-archs", []))
    info.rocm_archs = sorted(archs)
    if "rocm" in backends and framework in ("torch", "tvm-ffi"):
        rocm_sections = [
            (k, v)
            for k, v in cfg.get("kernel", {}).items()
            if isinstance(v, dict) and v.get("language") == "cpp-rocm"
        ]
        if not rocm_sections:
            err(
                "rocm-kernel",
                'rocm is a backend but no [kernel.*] section has language = "cpp-rocm"',
                toml_path,
            )
        for k, v in rocm_sections:
            if not v.get("rocm-archs"):
                err(
                    "rocm-archs",
                    f"[kernel.{k}] has no rocm-archs (the loader checks the GPU against them)",
                    toml_path,
                )
    if (
        "rocm" in backends
        and framework == "torch-noarch"
        and not cfg["torch-noarch"].get("rocm-archs")
    ):
        warn(
            "rocm-archs",
            "[torch-noarch] has no rocm-archs: the package will load on any ROCm GPU",
            toml_path,
        )

    package = name.replace("-", "_")
    info.package = package
    pkg_dir = config_dir / "torch-ext" / package
    if framework in ("torch", "torch-noarch"):
        if not (pkg_dir / "__init__.py").is_file():
            err(
                "package",
                f"Python package torch-ext/{package}/__init__.py is missing",
                pkg_dir,
            )
        else:
            findings += _check_imports(pkg_dir, package)
        findings += _check_namespaces(config_dir / "torch-ext")

    findings += _check_tests(config_dir / "tests", package)
    if (
        not (kernel_dir / "CARD.md").is_file()
        and not (kernel_dir / "README.md").is_file()
    ):
        warn("card", "no CARD.md or README.md for the Hub page", kernel_dir)
    if community:
        findings += _check_community(kernel_dir, name, repo_id, info.rocm_archs)
    return info, findings


def _check_imports(pkg_dir: Path, package: str) -> list[Finding]:
    out = []
    for py in sorted(pkg_dir.rglob("*.py")):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError as e:
            out.append(Finding("error", "syntax", f"does not parse: {e}", str(py)))
            continue
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module]
            for m in mods:
                if m.split(".")[0] == package:
                    out.append(
                        Finding(
                            "error",
                            "absolute-import",
                            f"line {node.lineno}: absolute self-import {m!r}; use a relative import "
                            "(the loader imports the package under its own module name)",
                            str(py),
                        )
                    )
    return out


def _check_namespaces(torch_ext: Path) -> list[Finding]:
    out = []
    if not torch_ext.is_dir():
        return out
    for path in sorted(torch_ext.rglob("*")):
        if (
            path.suffix in (".cpp", ".cc", ".cu", ".hip", ".h", ".hpp")
            and path.is_file()
        ):
            for i, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                code = line.split("//")[0]
                if FIXED_CPP_NS.search(code):
                    out.append(
                        Finding(
                            "error",
                            "op-namespace",
                            f"line {i}: fixed op namespace; use TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ...)",
                            str(path),
                        )
                    )
        elif path.suffix == ".py" and path.is_file():
            for i, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                if FIXED_PY_NS.search(line.split("#")[0]):
                    out.append(
                        Finding(
                            "error",
                            "op-namespace",
                            f"line {i}: fixed op namespace; use add_op_namespace_prefix(...) from ._ops",
                            str(path),
                        )
                    )
    return out


def _check_tests(tests: Path, package: str) -> list[Finding]:
    if not tests.is_dir():
        return [Finding("error", "tests", "tests/ is missing", str(tests))]
    texts = {
        p: p.read_text(encoding="utf-8", errors="replace")
        for p in sorted(tests.rglob("*.py"))
    }
    out = []
    # Look at the code, not the text, so a comment or docstring mentioning
    # get_kernel or kernels_ci doesn't count.
    calls_get_kernel = marks_kernels_ci = False
    for p, t in texts.items():
        try:
            tree = ast.parse(t, filename=str(p))
        except SyntaxError as e:
            out.append(Finding("error", "syntax", f"does not parse: {e}", str(p)))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and (
                getattr(node.func, "id", None) == "get_kernel"
                or getattr(node.func, "attr", None) == "get_kernel"
            ):
                calls_get_kernel = True
            elif isinstance(node, ast.Attribute) and node.attr == "kernels_ci":
                marks_kernels_ci = True
    if not calls_get_kernel:
        out.append(
            Finding(
                "error",
                "tests-get-kernel",
                "no test loads the kernel with get_kernel()",
                str(tests),
            )
        )
    if not marks_kernels_ci:
        out.append(
            Finding(
                "error",
                "tests-kernels-ci",
                "no test is marked kernels_ci (the subset CI runs)",
                str(tests),
            )
        )
    direct = re.compile(rf"^\s*(?:import\s+{package}\b|from\s+{package}\b)", re.M)
    for p, t in texts.items():
        if direct.search(t):
            out.append(
                Finding(
                    "warning",
                    "tests-direct-import",
                    f"imports {package} directly; load it with get_kernel()",
                    str(p),
                )
            )
    return out


def load_community_rules(path: Path = COMMUNITY_CONFIG) -> dict:
    rules = dict(COMMUNITY_DEFAULTS)
    if path.is_file():
        with open(path, "rb") as f:
            rules.update(tomllib.load(f))
    return rules


def _check_community(
    kernel_dir: Path, name: str, repo_id: str | None, rocm_archs: list[str]
) -> list[Finding]:
    rules = load_community_rules()
    out = []
    if rocm_archs and not set(rocm_archs) & set(rules["ci-archs"]):
        out.append(
            Finding(
                "error",
                "ci-arch",
                f"rocm-archs {rocm_archs} include none of the GPUs CI tests on "
                f"({', '.join(rules['ci-archs'])}), so its GPU tests can't run",
                str(kernel_dir),
            )
        )
    for prefix in rules["reserved-prefixes"]:
        if name.lower().startswith(prefix) or kernel_dir.name.lower().startswith(
            prefix
        ):
            out.append(
                Finding(
                    "error",
                    "reserved-name",
                    f"the {prefix}- prefix is reserved for AMD packages",
                    str(kernel_dir),
                )
            )
    if repo_id and REPO_ID.match(repo_id):
        org, repo_name = repo_id.split("/")
        if org.lower() in rules["reserved-orgs"]:
            out.append(
                Finding(
                    "error",
                    "reserved-org",
                    f"repo-id {repo_id!r} uses an AMD-reserved organisation",
                    str(kernel_dir),
                )
            )
        elif org != rules["hub-org"]:
            out.append(
                Finding(
                    "error",
                    "community-org",
                    f"repo-id {repo_id!r} must be in the {rules['hub-org']!r} organisation",
                    str(kernel_dir),
                )
            )
        if repo_name != kernel_dir.name:
            out.append(
                Finding(
                    "error",
                    "repo-name",
                    f"repo-id {repo_id!r} must be named after the directory: "
                    f"{rules['hub-org']}/{kernel_dir.name}",
                    str(kernel_dir),
                )
            )
    card = next(
        (
            kernel_dir / n
            for n in ("CARD.md", "README.md")
            if (kernel_dir / n).is_file()
        ),
        None,
    )
    notice = rules["card-notice"]
    if card and notice.lower() not in card.read_text(encoding="utf-8").lower():
        out.append(
            Finding(
                "error",
                "card-notice",
                f"{card.name} must say the kernel is {notice!r} "
                "(see docs/CARD_TEMPLATE_COMMUNITY.md)",
                str(card),
            )
        )
    if not any(kernel_dir.glob("LICENSE*")):
        out.append(
            Finding(
                "warning",
                "license-file",
                "no LICENSE file: the licence text should ship with the kernel",
                str(kernel_dir),
            )
        )
    for gm in (kernel_dir / ".gitmodules", kernel_dir.parent / ".gitmodules"):
        if gm.is_file():
            text = gm.read_text(encoding="utf-8", errors="replace")
            for block in re.split(r"\n(?=\[submodule)", text):
                path = re.search(r"path\s*=\s*(\S+)", block)
                url = re.search(r"url\s*=\s*(\S+)", block)
                if not path or not url or "aiter" not in url.group(1).lower():
                    continue
                target = (gm.parent / path.group(1)).resolve()
                if target == kernel_dir or kernel_dir in target.parents:
                    out.append(
                        Finding(
                            "error",
                            "aiter-code",
                            f"AITER submodule {path.group(1)}: AITER code belongs in ROCm/aiter",
                            str(gm),
                        )
                    )
    for sub in kernel_dir.rglob("aiter"):
        if sub.is_dir() and (
            (sub / ".git").exists() or (sub / "aiter" / "jit").is_dir()
        ):
            out.append(
                Finding(
                    "error",
                    "aiter-code",
                    "contains an AITER checkout; AITER code belongs in ROCm/aiter",
                    str(sub),
                )
            )
            break
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("kernels", nargs="+", help="kernel directory names (or paths)")
    ap.add_argument("--kernels-root", type=Path, default=Path.cwd())
    ap.add_argument(
        "--community",
        action="store_true",
        help="also apply the rocm-community-kernels rules",
    )
    ap.add_argument(
        "--require-backend",
        default="rocm",
        help="backend every kernel must build for ('' for none)",
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    results, failed = [], False
    for k in args.kernels:
        path = Path(k) if Path(k).is_dir() else args.kernels_root / k
        info, findings = check_kernel(
            path, community=args.community, require_backend=args.require_backend or None
        )
        failed |= any(f.level == "error" for f in findings)
        results.append(
            {"kernel": asdict(info), "findings": [asdict(f) for f in findings]}
        )
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            fs = r["findings"]
            errors = sum(f["level"] == "error" for f in fs)
            print(
                f"{r['kernel']['name']}: {errors} error(s), {len(fs) - errors} warning(s)"
            )
            for f in fs:
                where = f"  [{f['path']}]" if f["path"] else ""
                print(f"  {f['level'].upper():7} {f['code']}: {f['message']}{where}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
