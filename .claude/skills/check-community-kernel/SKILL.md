---
name: check-community-kernel
description: Check a kernel in ROCm/rocm-community-kernels against the interface and community rules CI enforces (build.toml, relative imports, build-generated op namespaces, get_kernel() tests with a kernels_ci subset, Hub repo id, card notice, maintainers), explain each finding in plain terms and fix it. Use before opening a pull request, when Kernel CI's Validate or ROCm job fails, or when reviewing a kernel pull request.
---

# Check a community kernel

Runs the checks CI runs, then explains and fixes what they find. This skill adds no
rules: [AGENTS.md](../../../AGENTS.md), [ci/community.toml](../../../ci/community.toml)
and the scripts in [scripts/](../../../scripts/README.md) are the source of truth, and
the commands below are the ones the workflows run.

## 1. Run the checks

```bash
python3 scripts/check_wrapping.py --community <kernel>      # interface and community rules
python3 scripts/kernel_maintainers.py check                 # maintainers registry
```

With Nix and an AMD GPU, also run what the MI350 job runs: `kernel-builder check-config`, one build variant, `check-abi`, and the `kernels_ci` tests:

```bash
python3 scripts/kernel_ci.py run --kernel <kernel> --ci --fail-on-skip
```

Logs and `summary.md` are in `ci-out/<run-id>/`. Without Nix or a GPU, say that the
build and GPU tests weren't run; CI runs them on the pull request.

To see what CI will check on a branch, list the kernels it changes:

```bash
python3 scripts/changed_kernels.py --base origin/main --head HEAD
```

## 2. Explain and fix each finding

Errors fail CI; warnings don't, but explain them. Fix the cause in the kernel, never by
changing `scripts/`, `ci/community.toml` or the workflows.

### build.toml and layout

| Code               | Meaning                                                                                                           | Fix                                                                           |
| ------------------ | ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `name`             | The directory name has characters outside `A-Z a-z 0-9 _ -`                                                       | Rename the directory                                                          |
| `flake`            | No `flake.nix`                                                                                                    | Use the template from the `add-community-kernel` skill, then `nix flake lock` |
| `build-toml`       | `build.toml` missing or not valid TOML                                                                            | Add it, or fix the syntax the message points to                               |
| `general-name`     | `[general] name` missing (error) or not the directory name (warning)                                              | Set it to the directory name                                                  |
| `edition`          | Not `edition = 6` (warning)                                                                                       | Set `edition = 6`; older files are upgraded with a warning                    |
| `version`          | `[general] version` missing or not a non-negative integer                                                         | Set an integer; it selects the Hub branch `v<version>`                        |
| `license`          | `[general] license` missing                                                                                       | Add the SPDX id of the kernel's licence                                       |
| `repo-id`          | `[general.hub] repo-id` missing or not `<org>/<name>`                                                             | `repo-id = "rocm-community/<kernel>"`                                         |
| `backends`         | `rocm` isn't in `[general] backends`                                                                              | Add `"rocm"`                                                                  |
| `framework`        | Not exactly one of `[torch]`, `[torch-noarch]`, `[tvm-ffi]`                                                       | `[torch]` for compiled kernels, `[torch-noarch]` for Triton / Python-only     |
| `rocm-kernel`      | `rocm` is a backend of a compiled kernel, but no `[kernel.*]` has `language = "cpp-rocm"`                         | Add a `cpp-rocm` section listing the HIP sources                              |
| `rocm-archs`       | A `cpp-rocm` section has no `rocm-archs`, or `[torch-noarch]` no `rocm_archs` (noarch: a warning in HF-only mode) | List the GPUs the kernel supports and was tested on; CI tests it on those     |
| `noarch-archs-key` | `[torch-noarch]` uses `rocm-archs`, which kernel-builder's check-config rejects                                   | Write `rocm_archs` (underscore) in that section                               |
| `card`             | No `CARD.md` or `README.md` (warning)                                                                             | Add `CARD.md` from `docs/CARD_TEMPLATE_COMMUNITY.md`                          |

### Python package and op registration

| Code              | Meaning                                                                                                          | Fix                                                                                                                                             |
| ----------------- | ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `package`         | No `torch-ext/<package>/__init__.py`; the package is the name with `-` replaced by `_`                           | Create or rename the package directory                                                                                                          |
| `syntax`          | A Python file in the package doesn't parse                                                                       | Fix the syntax error at the line given                                                                                                          |
| `absolute-import` | The package imports itself by name (`from my_kernel.x import y`)                                                 | Use a relative import (`from .x import y`): the loader imports the package under its own module name                                            |
| `op-namespace`    | An op is registered with a fixed namespace: `TORCH_LIBRARY(name, …)` in C++, or a literal `"name::op"` in Python | C++: `TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops)` (or `STABLE_TORCH_LIBRARY_EXPAND`). Python: `add_op_namespace_prefix("op")` from `._ops` |

### Tests

| Code                  | Meaning                                                       | Fix                                                                                                                                        |
| --------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `tests`               | No `tests/` directory                                         | Add tests (`*_test.py`)                                                                                                                    |
| `tests-get-kernel`    | No test loads the kernel with `get_kernel()`                  | `kernels.get_kernel("rocm-community/<kernel>", version=<version>)`, with build.toml's repo id and version; CI points it at the fresh build |
| `tests-kernels-ci`    | No test is marked `kernels_ci`, so CI's GPU job tests nothing | Mark a fast subset (under a minute in total) `@pytest.mark.kernels_ci`                                                                     |
| `tests-direct-import` | A test imports the package directly (warning)                 | Load it with `get_kernel()` instead, so tests exercise what users get                                                                      |

### Community rules

| Code            | Meaning                                                                            | Fix                                                                        |
| --------------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `community-org` | The repo id isn't in the community organisation (`hub-org` in `ci/community.toml`) | `repo-id = "rocm-community/<kernel>"`                                      |
| `repo-name`     | The repo id's name isn't the directory name                                        | Make them match                                                            |
| `reserved-name` | The name starts with a prefix reserved for AMD's packages (`aiter`)                | Rename. If it is AITER code, it belongs in ROCm/aiter, not here            |
| `reserved-org`  | The repo id uses `amd` or `amd-aiter`                                              | Use the community organisation                                             |
| `aiter-code`    | The kernel contains an AITER checkout or an AITER submodule                        | Remove it. Kernels here can't vendor AITER; AITER kernels go to ROCm/aiter |
| `ci-arch`       | `rocm-archs` includes none of the GPUs CI has a runner for (`[ci-runners]`)        | Test on and add one of them; otherwise CI can't run the kernel's tests     |
| `card-notice`   | `CARD.md` (or `README.md`) doesn't say the kernel is not supported by AMD          | Start from `docs/CARD_TEMPLATE_COMMUNITY.md`, which has the notice         |
| `license-file`  | No `LICENSE` file (warning)                                                        | Add the licence text; for a port, the original project's                   |

### Maintainers registry (`kernel_maintainers.py`)

| Message                                                               | Fix                                      |
| --------------------------------------------------------------------- | ---------------------------------------- |
| `<kernel>: missing from .github/kernel-maintainers.json`              | Add `"<kernel>": ["github-handle"]`      |
| `needs a list of at least one GitHub handle`                          | Add a maintainer; every kernel needs one |
| `not GitHub handles (write them without @)`                           | Remove the `@`                           |
| `in .github/kernel-maintainers.json but there is no kernel directory` | Remove the entry, or fix its spelling    |

### `kernel_ci.py` stages

| Result                                          | Look at        | Usual causes                                                                                                                   |
| ----------------------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| validate fails on `kernel-builder check-config` | `validate.log` | A key or section kernel-builder doesn't accept; a source in `src` that doesn't exist                                           |
| build fails                                     | `build.log`    | A source or header missing from `src`; HIP compile errors; a `rocm-archs` target the compiler doesn't know                     |
| `check-abi` fails                               | `build.log`    | The library uses symbols outside the Torch ABI it declares                                                                     |
| test fails                                      | `test.log`     | A `kernels_ci` test fails on the GPU, or `get_kernel()` can't find the build (check `flake.lock` pins a recent kernel-builder) |
| "skipped, not allowed here"                     | `summary.md`   | No GPU or Nix on the machine, or the GPU isn't in `rocm-archs`                                                                 |

## 3. Checks the scripts can't make

Look at these by hand and report them:

- **Version bump**: compare the public API (`__all__` in `torch-ext/<package>/__init__.py`
  and the ops in `torch_binding.cpp`) with `main`
  (`git diff origin/main -- <kernel>/torch-ext`). Any change, including an addition,
  needs `version` bumped in `build.toml` (AGENTS.md, "Kernel versions"), except as
  AGENTS.md lists.
- **Tested GPUs**: `rocm-archs` should only list GPUs the kernel was actually tested on.
- **Licence**: `LICENSE` matches `build.toml`'s `license`, and code from other projects
  keeps their notices.

## Rules

- Don't commit, push or open pull requests unless asked.
- Report results as they are, with the actual messages. Don't call a kernel compliant
  if a check couldn't run; say which ones didn't.
- After fixing, run the checks again and show they pass.
