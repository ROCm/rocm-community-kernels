# CI configuration

[`community.toml`](community.toml) holds the rules CI enforces for community kernels:
the Hub organisation, the reserved names and organisations, the GPUs CI tests on, and
the card notice. `scripts/check_wrapping.py --community` reads it, and so does the
`validate` stage of [Kernel CI](../.github/workflows/kernel-ci.yml).

The CI itself:

| Workflow                                                | When                      | What                                                                                                                                                                                             |
| ------------------------------------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [`kernel-ci.yml`](../.github/workflows/kernel-ci.yml)   | every pull request        | `scripts/kernel_ci.py` for each changed kernel: validate, then build one variant and run the `kernels_ci` tests on the MI350 ([`rocm-build-test.yml`](../.github/workflows/rocm-build-test.yml)) |
| [`publish.yml`](../.github/workflows/publish.yml)       | on request, from `main`   | build every variant and test it, approval, sign and upload, then `get_kernel()` from the Hub and test again                                                                                      |
| [`ci-scripts.yml`](../.github/workflows/ci-scripts.yml) | changes to the CI scripts | their unit tests                                                                                                                                                                                 |
| [`nix-fmt.yml`](../.github/workflows/nix-fmt.yml)       | changes to `.nix` files   | `nix fmt` check                                                                                                                                                                                  |

Run the pull request stages locally (Nix and an AMD GPU for build and test):

```bash
python3 scripts/kernel_ci.py run --kernel <kernel> --ci
```
