# scripts

The scripts the CI (the GitHub workflows) runs. Contributors can run them locally too.

| File                    | Purpose                                                                            | Used by                                              |
| ----------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------- |
| `kernel_ci.py`          | Runs the pull request stages for kernels: validate, build, GPU test                | `kernel-ci.yml`, `rocm-build-test.yml`, contributors |
| `check_wrapping.py`     | Checks a kernel is packaged correctly for kernel-builder, plus the community rules | `kernel_ci.py` (validate stage), contributors        |
| `ci_verify_load.py`     | Loads a kernel with `get_kernel()` as a user would, then runs its tests            | `publish.yml` (verify job)                           |
| `changed_kernels.py`    | Lists the kernels a pull request changes                                           | `kernel-ci.yml`                                      |
| `kernel_maintainers.py` | Checks the kernel maintainers registry                                             | `kernel-ci.yml`                                      |
| `publish_target.py`     | Works out where a publish uploads to, and how to load it back                      | `publish.yml`                                        |
| `publish_gate.py`       | Checks the publish approval gate (environment, reviewers, token) is in place       | `publish.yml` (target job)                           |
| `ci_download.py`        | Downloads a published kernel without loading it                                    | `publish.yml` (download job)                         |
| `tests/`                | Unit tests for the scripts above, and checks on the workflows                      | `kernel-ci.yml`, on every pull request               |

## `kernel_ci.py`

Runs three stages per kernel:

1. **validate**: `check_wrapping.py` (with the community rules unless
   `--hf-rules-only`), then `kernel-builder check-config`. kernel-builder's CLI comes
   from the repository's `flake.lock` (`nix run .#kernel-builder`); builds use each
   kernel's own `flake.lock`.
1. **build**: `nix build .#backendCi.rocm` (one representative variant, `--ci`) or
   `.#backendBundle.rocm` (every variant), then `kernel-builder check-abi` for compiled
   kernels. Memory-heavy kernels get Nix job limits from `build-concurrency.json`.
1. **test**: builds the GPU test for the newest Torch with ROCm 7.1
   (`.#ciTests.<variant>`, e.g. `torch213-rocm71-x86_64-linux`) and runs it on the AMD
   GPU, which runs the kernel's `kernels_ci` tests. Kernel-builder's own `.#ci-test`
   picks ROCm 7.14, whose toolchain isn't in Hugging Face's binary cache.
   `--ci-test-rocm` picks another ROCm version; `--ci-test-rocm ''` uses `.#ci-test`.
   A kernel with no ROCm 7.1 variant falls back to `.#ci-test`, with a warning.

Writes `summary.md` and `summary.json`, and per kernel `findings.json`, logs and the
build, to `ci-out/<run-id>/`.

- `--fail-on-skip` makes a stage that can't run (no GPU, no Nix, or a GPU the kernel
  doesn't target) fail instead of skip. The workflows use it, so a gate never passes
  without running.
- Commands never get a Hugging Face token: the GPU runner runs pull request code.
- `build` and `test` need Nix to use Hugging Face's binary cache
  (`nix run nixpkgs#cachix -- use huggingface`). They fail early without it
  (`--no-hf-cache-check` overrides this). They warn, before building, about toolchain
  derivations (LLVM, clang, rocm-clr, PyTorch) missing from the cache, which Nix would
  compile from source, and list them in `summary.json` under `toolchain_from_source`.

```bash
python3 scripts/kernel_ci.py run --kernel <kernel> --ci          # Nix and an AMD GPU
python3 scripts/kernel_ci.py run --kernel <kernel> --dry-run     # show the commands
```

## `check_wrapping.py`

Reports errors and warnings for each kernel; exits 1 if there's an error.

- **Hugging Face's rules**: `flake.nix` and `build.toml` (name, version, licence, repo
  id); ROCm kernel sections list `rocm-archs`; the Python package imports its own
  modules relatively; Torch ops use the build-generated namespace; tests load the
  kernel with `get_kernel()` and mark a fast subset `kernels_ci`.
- **Community rules** (`--community`), from [`ci/community.toml`](../ci/community.toml):
  repo id `rocm-community/<directory name>`; no `aiter-` names, no `amd` or `amd-aiter`
  organisations, no AITER code; the card says "not supported by AMD"; `rocm-archs`
  includes a GPU CI tests on (gfx950); a `LICENSE` file (warning only).

```bash
python3 scripts/check_wrapping.py --community <kernel>
```

## `ci_verify_load.py`

Runs in a plain user environment (PyTorch for ROCm and the `kernels` library, no Nix).
Loads the kernel with `get_kernel()`, records the build variant the loader chose and
any digest or signature warnings, then optionally runs the kernel's `kernels_ci` tests
(`--tests`), a smoke expression (`--smoke`) or its benchmarks (`--benchmark`). Writes a
JSON report; exits 1 on any failure.

- `--tests-repo-id`: the publish workflow can upload to another organisation, but the
  kernel's tests load `build.toml`'s repo id. This maps that id, through
  `LOCAL_KERNELS`, to the build just downloaded, so the tests run against the published
  build.
- `--expect-under`: fails unless the kernel loaded from below a given directory.
- `--benchmark`: times the `Benchmark` classes in the kernel's
  `benchmarks/benchmark.py` and records the results in the report.

## `publish_gate.py`

Runs first in the publish workflow. Fails unless the `hf-rocm-community` environment
exists with required reviewers and a branch limit, and `HF_TOKEN` is a secret of that
environment only (a repository or organisation secret would let the upload run without
approval). See Repository setup in [`ci/README.md`](../ci/README.md).

## `changed_kernels.py`

Prints the changed kernels as JSON (with `--github-output`, also for the workflow). A
kernel is a top-level directory with `flake.nix` and `build.toml` (or `src/build.toml`).
`--kernels a,b` or `--kernels all` names them instead, for manual runs. Names outside
`[A-Za-z0-9_-]` are rejected, because they end up in workflow commands.

```bash
python3 scripts/changed_kernels.py --base origin/main --head HEAD
```

## `kernel_maintainers.py`

Checks [`.github/kernel-maintainers.json`](../.github/kernel-maintainers.json): every
kernel directory is listed, every entry has at least one GitHub handle (without `@`),
and no entry points to a missing directory. Kernel maintainers are not CODEOWNERS:
GitHub only accepts code owners with write access to the repository.

```bash
python3 scripts/kernel_maintainers.py check
```

## `publish_target.py`

Combines the organisation from the publish workflow (its `hub_org` input, or the
`HUB_ORG` repository variable) with the name and version in `build.toml`. Outputs the
upload repo id and branch (`v<version>`, plus `main` for version 1), `build.toml`'s own repo id (for `--tests-repo-id`), the arguments to load the
branch back, and the kernel's config directory. Refuses AMD-reserved organisations and
unsafe branch names.

```bash
python3 scripts/publish_target.py <kernel> --hub-org <test-org>
```

## `tests/`

No GPU, Nix or network needed.

| File                     | Covers                                                                                                                                                                                                               |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `check_wrapping_test.py` | Each wrapping and community rule, on small generated kernels                                                                                                                                                         |
| `kernel_ci_test.py`      | GPU-architecture skipping, `--fail-on-skip`, build concurrency, expected results, no Hub token in commands                                                                                                           |
| `repo_tooling_test.py`   | `changed_kernels.py` on a real git history, the maintainers check, `publish_target.py`, `publish_gate.py`; the workflows: uploads are private, GPU jobs run in a container with no secrets, kernel-builder is pinned |
| `ci_verify_load_test.py` | `--tests-repo-id` with the real `kernels` loader (skipped without PyTorch and `kernels`)                                                                                                                             |

```bash
python3 -m pytest scripts/tests
```
