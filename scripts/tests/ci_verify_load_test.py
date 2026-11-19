"""ci_verify_load.py with the real `kernels` loader, as publish.yml's verify job runs it.

The kernel is loaded through LOCAL_KERNELS instead of the Hub, so no network or
GPU is needed. Skipped where torch or kernels isn't installed.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("kernels")
pytest.importorskip("torch")

SCRIPTS = Path(__file__).resolve().parents[1]

KERNEL_TEST = """\
import kernels
import pytest
import torch

demo = kernels.get_kernel("rocm-community/demo", version=1)


@pytest.mark.kernels_ci
def test_double():
    assert torch.equal(demo.double(torch.tensor([1.0, 2.0])), torch.tensor([2.0, 4.0]))
"""
CONFTEST = (
    "def pytest_configure(config):\n"
    "    config.addinivalue_line('markers', 'kernels_ci: CI subset')\n"
)


def make_build(root: Path) -> Path:
    """A Python-only build variant for this machine, with a real digest."""
    from kernels._rust import Digest, DigestAlgorithm
    from kernels.backends import _backend

    backend = _backend().name
    variant = root / "build" / f"torch-{backend}"
    variant.mkdir(parents=True)
    (variant / "__init__.py").write_text("def double(x):\n    return x * 2\n")
    digest = Digest.hash_variant(DigestAlgorithm.SHA256, variant)
    metadata = {
        "name": "demo",
        "id": "demo_test",
        "version": 1,
        "license": "Apache-2.0",
        "python-depends": [],
        "backend": {"type": backend},
        "digest": {"algorithm": "sha256", "files": digest.files},
    }
    (variant / "metadata.json").write_text(json.dumps(metadata))
    return root


def test_kernel_tests_follow_a_publish_to_another_org(tmp_path):
    """publish.yml uploads to a test organisation, while the tests load build.toml's id."""
    published = make_build(tmp_path / "test-org-demo")
    tests = tmp_path / "kernel" / "tests"
    tests.mkdir(parents=True)
    (tests / "double_test.py").write_text(KERNEL_TEST)
    (tests / "conftest.py").write_text(CONFTEST)

    def verify(*extra: str) -> tuple[int, dict]:
        out = tmp_path / f"report-{len(list(tmp_path.glob('report-*')))}.json"
        env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        # Stands in for the download from the Hub: the published repo resolves locally.
        env.update(
            LOCAL_KERNELS=f"test-org/demo={published}",
            HF_HUB_OFFLINE="1",
            HF_HOME=str(tmp_path / "hf-home"),
        )
        cmd = [sys.executable, str(SCRIPTS / "ci_verify_load.py")]
        cmd += ["--repo-id", "test-org/demo", "--version", "1", "--json", str(out)]
        cmd += ["--tests", str(tests), *extra]
        rc = subprocess.run(cmd, env=env, cwd=tmp_path, capture_output=True).returncode
        return rc, json.loads(out.read_text())

    rc, report = verify()
    assert rc == 1, "the tests load rocm-community/demo, which isn't available"
    rc, report = verify("--tests-repo-id", "rocm-community/demo")
    assert rc == 0, report
    assert report["tests_exit"] == 0
    assert report["tests_local_kernels"].startswith("rocm-community/demo=")
