---
name: add-community-kernel
description: Scaffold a new community kernel in ROCm/rocm-community-kernels with the correct interface for kernel-builder and the Hugging Face Kernel Hub (build.toml, flake.nix, Torch op binding or Triton package, get_kernel() tests with a kernels_ci subset, card, maintainers), or port an existing kernel into that layout. Use when a contributor wants to add a kernel, port one from elsewhere, or asks how a kernel here must be laid out.
---

# Add a community kernel

Creates `<kernel>/` with the interface every kernel here needs, from the templates in
`templates/`. The rules come from the repository, not from this skill:
[AGENTS.md](../../../AGENTS.md) (packaging rules),
[ci/community.toml](../../../ci/community.toml) (community rules) and
[CONTRIBUTING.md](../../../CONTRIBUTING.md). Passing this skill's final check means the
CI's validate stage passes too, because it runs the same script.

## Before starting: refuse what doesn't belong here

Stop and explain, without scaffolding anything, if:

- the kernel is AITER code, a copy or fork of an AITER kernel, or depends on an AITER
  checkout or submodule. AMD's kernels go to [ROCm/aiter](https://github.com/ROCm/aiter);
- the user wants a name starting `aiter`, or the `amd` or `amd-aiter` Hub organisation;
- the licence doesn't allow redistributing built binaries, or nobody will maintain it.

## 1. Ask for what you can't infer

- **Name**: lower case with hyphens (`my-kernel`). It is the directory, `[general] name`
  and the Hub repository name. The Python package is the same with underscores
  (`my_kernel`).
- **Kind**:
  - **Triton** (Python only, `[torch-noarch]`): one build for every PyTorch and ROCm
    version; Triton compiles on the user's GPU on first use. Start from
    `templates/triton/`.
  - **HIP / C++** (compiled, `[torch]`): one build per PyTorch and ROCm version, for
    the GPUs in `rocm-archs`. Start from `templates/hip/`.
- **Licence**: an SPDX id for `build.toml`, plus the licence text for `LICENSE`. For a
  port, the original project's licence.
- **Maintainers**: GitHub handles, at least one.
- **Source**: where the code comes from, for the README and the card.

## 2. Create the files

Copy the template for the kind, then the common files:

```bash
K=my-kernel                              # the new kernel's name
T=.claude/skills/add-community-kernel/templates
mkdir "$K"
cp -r "$T/triton/." "$K/"                # or "$T/hip/." for a compiled kernel
cp "$T/common/flake.nix" "$T/common/README.md" "$K/"
cp docs/CARD_TEMPLATE_COMMUNITY.md "$K/CARD.md"
mv "$K/torch-ext/my_kernel" "$K/torch-ext/${K//-/_}"
```

Then replace every `my-kernel` with the name and every `my_kernel` with the package
name, in all the copied files including `tests/my_kernel_test.py` (rename it
`<package>_test.py`). Set `license` in `build.toml`, add `LICENSE`, and fill in the
README.

Resulting layout (compiled kernel; a Triton kernel has no `torch-ext/torch_binding.*`
or `<package>_hip/`):

```text
my-kernel/
├── build.toml                 name, version, licence, repo id, sources, rocm-archs
├── flake.nix                  pins kernel-builder; run `nix flake lock` to create flake.lock
├── CARD.md                    the Hub page; keeps the "not supported by AMD" notice
├── README.md
├── LICENSE
├── my_kernel_hip/scale.hip    the kernel (compiled only)
├── torch-ext/
│   ├── torch_binding.cpp      op registration (compiled only)
│   ├── torch_binding.h
│   └── my_kernel/__init__.py  the public Python API
└── tests/my_kernel_test.py    loads the kernel with get_kernel(); kernels_ci subset
```

## 3. Replace the example with the real kernel, keeping the interface

The templates implement `scale(x, factor)` as a working example. When replacing it,
keep these, which CI checks (`scripts/check_wrapping.py`):

- **Relative imports** inside `torch-ext/<package>/`: `from ._scale import scale`,
  never `from my_kernel._scale import scale`.
- **Op names from the build** (compiled kernels): register in C++ with
  `TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops)`; in Python, call `ops.<name>` from
  `._ops`, and wrap names passed to `torch.library` functions such as `register_fake` or
  `custom_op` in `add_op_namespace_prefix(...)`. Never hard-code a namespace.
- **ROCm dispatch**: implement ops for `torch::kCUDA`; ROCm builds of PyTorch use the
  CUDA dispatch key, and tensors are on device `"cuda"`.
- **Sources listed in `build.toml`**: every `.hip`, `.cpp` and header the build needs
  goes in `src` of a `[kernel.*]` section with `language = "cpp-rocm"`, and only those
  files are copied into the build.
- **`rocm-archs`** (compiled kernels) lists only GPUs the kernel has been tested on,
  and includes `gfx950`, which CI tests on.
- **Tests** load the kernel with
  `kernels.get_kernel("rocm-community/<name>", version=<version>)` and never import the
  package. Mark a fast subset (under a minute in total) `@pytest.mark.kernels_ci`; that
  is what CI runs on the GPU. Test files are named `*_test.py`.
- **The public API** is what `__init__.py` exports (`__all__`). Changing it later,
  including adding to it, needs a `version` bump (AGENTS.md, "Kernel versions").

For a compiled kernel, consider the Torch stable ABI (`[torch.stable-abi]`, AGENTS.md):
one build then serves every PyTorch from that version on. It changes the binding and
the tensor API (`torch::stable::Tensor`, `STABLE_TORCH_LIBRARY_EXPAND`, `TORCH_BOX`).
The template uses the full Torch API, the simpler starting point for a ROCm-only
kernel.

## 4. Register the maintainers

Add the kernel to `.github/kernel-maintainers.json`, keeping the keys sorted:

```json
{
  "my-kernel": ["github-handle"]
}
```

## 5. Check it

```bash
python3 scripts/check_wrapping.py --community <kernel>     # no errors allowed
python3 scripts/kernel_maintainers.py check
python3 scripts/kernel_ci.py run --kernel <kernel> --ci    # with Nix and an AMD GPU
```

Use the `check-community-kernel` skill to explain and fix any finding, then the
`lint-community-kernels` skill before committing. A `license-file` warning means
`LICENSE` is still missing. On a Triton kernel, the `rocm-archs` warning is expected:
the package loads on any ROCm GPU.

## Rules

- Don't commit, push or open a pull request unless the user asks.
- Don't touch other kernels, `scripts/`, `ci/` or the workflows.
- Report check results as they are. If you couldn't run the build or the GPU tests (no
  Nix or no AMD GPU), say so; CI will run them on the pull request.
