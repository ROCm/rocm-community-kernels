# rocm-community-kernels

Community GPU kernels for ROCm, built and tested on AMD GPUs and published to the
[Hugging Face Kernel Hub](https://huggingface.co/docs/kernels/index).

> [!IMPORTANT]
> Kernels in this repository are written and maintained by their contributors. AMD
> maintainers review every change for policy and safety, and AMD runs the CI that
> builds, tests and publishes them, but **these kernels are not supported by AMD**.
> Report problems with a kernel to its maintainers (see
> [`.github/kernel-maintainers.json`](.github/kernel-maintainers.json)). AMD's own
> kernels are published from [ROCm/aiter](https://github.com/ROCm/aiter).

## Status

Proof of concept. The repository, its review rules and its CI are being set up and
checked on MI350 (gfx950); see [docs/poc/goal1.md](docs/poc/goal1.md). No kernels have
been published from here yet, and the Hub organisation is not final.

## Using a kernel

Users never build anything: the [`kernels`](https://github.com/huggingface/kernels)
library downloads the build that matches your PyTorch, ROCm version and GPU.

```python
# pip install kernels, plus PyTorch for ROCm
import torch
from kernels import get_kernel

my_kernel = get_kernel("rocm-community/<kernel>", version=1)
```

Pin a `version`: it selects the Hub branch `v<version>`, which only changes through a
reviewed release of that kernel.

## Repository layout

One directory per kernel, each built with Hugging Face's
[kernel-builder](https://github.com/huggingface/kernels) and published as one Hub
repository.

```text
<kernel>/                 one community kernel, published as rocm-community/<kernel>
├── build.toml            kernel-builder configuration: name, version, licence, repo-id, sources
├── flake.nix, flake.lock pins kernel-builder
├── CARD.md               the Hub page (from docs/CARD_TEMPLATE_COMMUNITY.md)
├── LICENSE
├── torch-ext/            the Python package and Torch op bindings
└── tests/                tests that load the kernel with get_kernel(); a kernels_ci subset runs in CI
AGENTS.md                 the rules every kernel follows (imports, op namespaces, tests, versions)
CONTRIBUTING.md           how to add or change a kernel
docs/                     community policy, card template, POC runbook
ci/                       the community rules CI enforces (community.toml)
scripts/                  the checks CI runs (kernel_ci.py runs them locally too)
.github/                  workflows, CODEOWNERS, maintainers registry, branch ruleset
```

## How a kernel gets to users

1. **Pull request.** CI checks how the kernel is wrapped and the community rules,
   builds one ROCm variant with kernel-builder, and runs its `kernels_ci` tests on an
   AMD GPU ([kernel-ci.yml](.github/workflows/kernel-ci.yml)). An AMD maintainer must
   approve, and CI must pass, before it merges.
1. **Publish, on request.** When a kernel's maintainers ask for a release, an AMD
   maintainer runs [publish.yml](.github/workflows/publish.yml). It builds every
   variant, runs the GPU tests, and waits for approval before it signs the build and
   uploads it to the Hub. It then loads the kernel back with `get_kernel()` on a clean
   machine and runs its tests again.

Nothing is published on a schedule or automatically on merge.

To run the same stages on your own machine (Nix and an AMD GPU for build and test):

```bash
python3 scripts/check_wrapping.py --community <kernel>
python3 scripts/kernel_ci.py run --kernel <kernel> --ci
```

## Contributing

Contributions are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) and
[`docs/COMMUNITY_POLICY.md`](docs/COMMUNITY_POLICY.md) before opening a pull request.

## Security

Please follow [SECURITY.md](SECURITY.md) for reporting vulnerabilities.

## Maintainers

| Role                       | Contact                                    |
| -------------------------- | ------------------------------------------ |
| AMD repository maintainers | `@ROCm/rocm-community-kernels-maintainers` |
| Kernel maintainers         | `.github/kernel-maintainers.json`          |

## License

The repository's own files (CI, scripts, documentation) are licensed under the terms in
the [LICENSE](LICENSE) file. Each kernel carries its own licence, declared in its
`build.toml` and shipped with it.
