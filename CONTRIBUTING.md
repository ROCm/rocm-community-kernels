# Contributing to rocm-community-kernels

This repository hosts community GPU kernels for ROCm and publishes them to the Hugging
Face Kernel Hub. Every kernel is built with kernel-builder and follows the packaging
rules in [AGENTS.md](AGENTS.md).

______________________________________________________________________

> **Security vulnerabilities**: do not open a public GitHub issue. See
> [SECURITY.md](SECURITY.md) for the private reporting process.

______________________________________________________________________

## Which kernels are accepted?

Kernels that are useful to ROCm users and whose authors want them published for AMD
GPUs: for example hackathon or GPU MODE entries, research kernels, or ROCm ports of
kernels that only exist for other hardware. The full policy, including maintainer
duties and when a kernel is removed, is in
[docs/COMMUNITY_POLICY.md](docs/COMMUNITY_POLICY.md). In short:

- **The kernel builds for ROCm** with kernel-builder and its tests pass on an AMD GPU.
- **It has named maintainers** who answer issues about it. AMD does not support
  community kernels.
- **Its licence allows publishing.** The licence is declared in `build.toml` and its
  text ships with the kernel.
- **It is not AITER code.** Kernels from [ROCm/aiter](https://github.com/ROCm/aiter),
  and changes to them, go to AITER, which publishes AMD's own kernels to `amd/*`. The
  `aiter-` name prefix and the `amd` and `amd-aiter` Hub organisations are reserved.

## How to add a kernel

Each kernel is one top-level directory, published as one Hub repository. The
`add-community-kernel` skill (below) scaffolds one; the
[Kernel Hub documentation](https://huggingface.co/docs/kernels/index) covers writing and
building kernels with kernel-builder.

1. Create `<kernel>/` with a `flake.nix` that pins kernel-builder (the skill's template
   has one) and a `build.toml`:
   - `[general]`: `name` (the directory name), `version = 1`, `license`,
     `backends = ["rocm"]` (more backends are fine);
   - `[general.hub]`: `repo-id = "rocm-community/<kernel>"`;
   - for compiled kernels, a `[kernel.*]` section with `language = "cpp-rocm"` and the
     `rocm-archs` you have tested. Include `gfx950`; CI tests on MI350.
1. Add the Python package in `torch-ext/<kernel_name>/` and, for compiled kernels, the
   Torch op binding in `torch-ext/`. Follow [AGENTS.md](AGENTS.md): relative imports,
   op names from the build-generated namespace.
1. Add `tests/`. Tests load the kernel with `get_kernel()`, never by importing it, and
   a fast subset (under a minute) is marked `@pytest.mark.kernels_ci`. Test files are
   named `*_test.py`.
1. Add `CARD.md` from [docs/CARD_TEMPLATE_COMMUNITY.md](docs/CARD_TEMPLATE_COMMUNITY.md)
   (it keeps the "not supported by AMD" notice), a `README.md` with the link to the
   original source, and the licence text as `LICENSE`.
1. Add yourself as the kernel's maintainer in
   [.github/kernel-maintainers.json](.github/kernel-maintainers.json): the kernel's
   directory name and a list of GitHub handles, without `@`.
1. Check it locally:
   ```bash
   python3 scripts/check_wrapping.py --community <kernel>
   python3 scripts/kernel_maintainers.py check
   # with Nix and an AMD GPU: build one variant and run the kernels_ci tests
   python3 scripts/kernel_ci.py run --kernel <kernel> --ci
   ```

Do not commit build outputs (`build/`, `result*`, `*.so`).

### With a coding agent

The repository ships [Claude Code skills](.claude/skills/) that walk through these
steps and run the same checks as CI:

| Skill                    | Use it to                                                                                                                                                              |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `add-community-kernel`   | Scaffold a new kernel, or port one, with the right interface: `build.toml`, `flake.nix`, the Triton package or HIP op binding, `get_kernel()` tests, card, maintainers |
| `check-community-kernel` | Run CI's checks on a kernel, explain each finding and fix it                                                                                                           |
| `lint-community-kernels` | Run the pre-commit hooks exactly as CI does, and fix formatting                                                                                                        |

## Pull requests

Title pull requests `<kernel>: short description`, for example
`my-kernel: add bf16 support`, and change one kernel per pull request.

Every pull request runs [Kernel CI](.github/workflows/kernel-ci.yml) for each kernel it
changes:

- **Validate**: the maintainers registry, how the kernel is wrapped
  (`scripts/check_wrapping.py --community`), and `kernel-builder check-config`.
- **Build and GPU test**: one representative ROCm variant with kernel-builder, then the
  `kernels_ci` tests on an MI350, loading the fresh build through `get_kernel()`.

Initially, CI tests on an MI350 only; more hardware variants will be added shortly.

For first-time contributors, a maintainer approves the workflow run before it starts.
To merge, a pull request needs "Kernel CI result" and the pre-commit checks to pass and
an approval from an AMD repository maintainer (the code owners). Changes to a kernel
also need its maintainers' agreement on the pull request; AMD reviewers check for it.
AMD's review checks the community rules and safety; it is not a quality or support
commitment.

Bump `version` in `build.toml` for any change to the public API, additions included
([AGENTS.md](AGENTS.md#kernel-versions)).

## Publishing

Nothing is published automatically. When a kernel is ready, its maintainers ask for a
release in an issue or on their pull request. An AMD maintainer then runs
[Publish kernel](.github/workflows/publish.yml). It builds every variant on an AMD GPU
and runs the tests, then waits for an AMD maintainer to approve the upload. After
approval it signs each variant and uploads it to the Hub, to branch `v<version>`. It
then loads the kernel back from the Hub with `get_kernel()` on a clean machine and runs
its `kernels_ci` tests.

## Developer policies

> [!IMPORTANT]
> AMD employees must also follow the ROCm open source software contributing policies at
> http://u.amd.com/rocm-oss-policies.

This project is covered by the
[ROCm Project Governance](https://github.com/ROCm/ROCm/blob/develop/GOVERNANCE.md),
which also defines the code of conduct.

The repository's own files are covered by the [LICENSE](LICENSE) file. Each kernel is
covered by the licence declared in its `build.toml`.

### Code style

The pre-commit hooks (`.pre-commit-config.yaml`) run on every pull request: black for
Python, mdformat for Markdown, actionlint for workflows, and `*_test.py` test naming.
Run them locally with `pre-commit run --all-files`. Otherwise follow the
[ROCm/TheRock style guides](https://github.com/ROCm/TheRock/tree/main/docs/development/style_guides).

### Security requirements

Contributors must not:

- Commit secrets, tokens, passwords, or credentials.
- Introduce vulnerable dependencies without justification.
- Bypass security controls or required security reviews.

All contributions are subject to automated security scanning.
