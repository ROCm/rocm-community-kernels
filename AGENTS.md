# Rules for kernels in this repository

These rules are for contributors and for coding agents working on kernels here. The
packaging rules make a kernel build with kernel-builder and load correctly with
`get_kernel()`; the community rules set what this repository accepts. CI checks most of
them with `python3 scripts/check_wrapping.py --community <kernel>`.

Skills in `.claude/skills/` apply these rules: `add-community-kernel` to create or port
a kernel, `check-community-kernel` to check one and fix findings, and
`lint-community-kernels` before committing.

# Packaging rules

## Imports

Inside a kernel's package (`torch-ext/<kernel_name>/`), modules of the same kernel
import each other relatively. The loader imports the package under its own module
name, so an absolute self-import fails or picks up the wrong copy.

```python
# Wrong
from my_kernel.utils import pack

# Right
from .utils import pack
```

## The `_ops` module and op registration

The build generates `torch-ext/<kernel_name>/_ops.py`. It provides `ops` (the kernel's
Torch ops) and `add_op_namespace_prefix(name)`, which returns the op name in the
kernel's own namespace. That namespace is unique per build, so that two kernels, or two
versions of one kernel, can be loaded side by side. Never hard-code a namespace:

- in Python, wrap every op name passed to `torch.library.custom_op`, `register_fake`,
  `register_kernel` and similar in `add_op_namespace_prefix(...)`;
- in C++, register with `TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops)` (or the
  stable-ABI equivalent), never `TORCH_LIBRARY(my_namespace, ...)`.

```python
# Wrong
@torch.library.custom_op("my_kernel::gemm", mutates_args=())
def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...


# Right
@torch.library.custom_op(add_op_namespace_prefix("gemm"), mutates_args=())
def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...
```

## Tests

Tests live in `tests/` and load the kernel the way users do, with `get_kernel()`, using
the `repo-id` and `version` from `build.toml`. Never import the built package directly.

```python
import kernels

my_kernel = kernels.get_kernel("rocm-community/my-kernel", version=1)
```

In CI, `LOCAL_KERNELS` points `get_kernel()` at the fresh build, so the tests exercise
the loader, variant selection and the package as published. Mark a fast subset (under
a minute in total) with `@pytest.mark.kernels_ci`; that subset is what
the GPU test runs (`nix run .#ciTests.<variant>`, or `.#ci-test` for kernel-builder's
default variant). In this repository test files are named
`*_test.py`.

## Kernel versions

`version` in `build.toml` selects the Hub branch (`v<version>`) that users pin. Bump it
for any change to the public API, including additions such as a new function, layer or
op. Kernel-builder only rebuilds the newest Torch versions, so older variants on the
same branch keep the old API; an addition without a bump would advertise functions that
some variants don't have. Appending optional parameters doesn't need a bump, and
neither do additive changes where every published variant is rebuilt anyway:
Python-only (`[torch-noarch]`) kernels and kernels on the Torch stable ABI.

## Torch stable ABI

Prefer the Torch stable ABI for compiled kernels (`[torch.stable-abi]` in `build.toml`,
for example `rocm = "2.10"`, the lowest version that works). One build then serves
every Torch version from that one onwards, instead of a build per Torch version. Guard
code that needs newer APIs with `TORCH_FEATURE_VERSION`.

# Community rules (this repository only)

These are configured in [ci/community.toml](ci/community.toml) and described in
[docs/COMMUNITY_POLICY.md](docs/COMMUNITY_POLICY.md).

- **Hub repository.** `repo-id` is `rocm-community/<kernel directory name>`. The
  organisation name is proposed, not yet agreed with Hugging Face. The publish workflow
  can publish to another organisation (its `hub_org` input) without changing
  `build.toml`.
- **Reserved names.** No `aiter-` prefix, and no `amd` or `amd-aiter` organisation.
  Those belong to AMD's own kernels.
- **No AITER code.** Don't vendor AITER sources, add AITER as a submodule, or submit
  AITER kernels or changes to them.
- **Card notice.** `CARD.md` (or `README.md`) says the kernel is not supported by AMD;
  start from [docs/CARD_TEMPLATE_COMMUNITY.md](docs/CARD_TEMPLATE_COMMUNITY.md).
- **Licence.** `license` in `build.toml`, and the licence text in the kernel's
  directory.
- **Maintainers.** Every kernel has at least one maintainer (a GitHub handle) in
  `.github/kernel-maintainers.json`.
- **Architectures.** `rocm-archs` lists only GPUs the kernel has been tested on, and
  includes `gfx950`, which CI tests on.
