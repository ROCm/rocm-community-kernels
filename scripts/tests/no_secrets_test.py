"""Guard: no token or private key in any file of the repository.

A local backstop to the gitleaks scan that runs on every pull request. Checks
every tracked file and every untracked file git doesn't ignore. Tokens are only
ever supplied at run time: GitHub secrets referenced by name in the workflows,
or HF_TOKEN in the environment of whoever runs a script.
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

PATTERNS = {
    "Hugging Face token": r"\bhf_[A-Za-z0-9]{30,}",
    "GitHub token": r"\bgh[pousr]_[A-Za-z0-9]{36,}",
    "GitHub fine-grained token": r"\bgithub_pat_[A-Za-z0-9_]{50,}",
    "AWS access key": r"\bAKIA[0-9A-Z]{16}\b",
    "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
}


def repo_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "-z", "--cached", "--others"]
            + ["--exclude-standard"],
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [REPO_ROOT / p for p in out.decode().split("\0") if p]


def test_no_tokens_in_the_repository():
    files = repo_files()
    assert files
    found = []
    for path in files:
        if not path.is_file():  # deleted but not yet committed
            continue
        text = path.read_bytes().decode(errors="ignore")
        for kind, pattern in PATTERNS.items():
            for m in re.finditer(pattern, text):
                line = text.count("\n", 0, m.start()) + 1
                found.append(f"{path.relative_to(REPO_ROOT)}:{line}: {kind}")
    assert not found, "token-like strings (revoke them, then remove):\n" + "\n".join(
        found
    )
