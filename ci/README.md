# CI configuration

[`community.toml`](community.toml) holds the rules CI enforces for community kernels:
the Hub organisation, the reserved names and organisations, the GPUs CI tests on, and
the card notice. `scripts/check_wrapping.py --community` reads it, and so does the
`validate` stage of [Kernel CI](../.github/workflows/kernel-ci.yml).

The CI itself:

| Workflow                                              | When                    | What                                                                                                                                                                                                                             |
| ----------------------------------------------------- | ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`kernel-ci.yml`](../.github/workflows/kernel-ci.yml) | every pull request      | the CI scripts' unit tests, and `scripts/kernel_ci.py` for each changed kernel: validate, then build one variant and run the `kernels_ci` tests on the MI350 ([`rocm-build-test.yml`](../.github/workflows/rocm-build-test.yml)) |
| [`publish.yml`](../.github/workflows/publish.yml)     | on request, from `main` | build every variant and test it, approval, sign and upload, then `get_kernel()` from the Hub and test again                                                                                                                      |
| [`ci-image.yml`](../.github/workflows/ci-image.yml)   | changes to `ci/docker/` | build the CI image; from `main`, push it as `latest`, the tag the GPU jobs run                                                                                                                                                   |
| [`nix-fmt.yml`](../.github/workflows/nix-fmt.yml)     | changes to `.nix` files | `nix fmt` check                                                                                                                                                                                                                  |

Run the pull request stages locally (Nix and an AMD GPU for build and test):

```bash
python3 scripts/kernel_ci.py run --kernel <kernel> --ci
```

## Repository setup

What an administrator sets up outside the repository.

First, for both pull requests and publishing: set the `GPU_TEST_RUNNER` variable
(repository or organisation level) to the label of the GPU test runner, an MI350
(gfx950) runner with one GPU. The GPU jobs run on it, and the workflows name no runner
of their own, so moving to another runner needs no change here. Kernel CI's `validate`
job and the publish workflow fail without it. It's configuration, not a secret.

### Publishing

In this order, before `HF_TOKEN` exists anywhere and before `publish.yml` reaches
`main`. The publish workflow checks steps 1 and 2 before it builds anything
(`scripts/publish_gate.py`), and refuses to run without them.

1. Create the `hf-rocm-community` environment, with:
   - required reviewers: the AMD repository maintainers
     (`@ROCm/rocm-community-kernels-maintainers`);
   - deployment branches: selected branches, `main` only.
1. Add `HF_TOKEN`, a Hub write token for the organisation, as a secret of that
   environment only. A repository or organisation secret is visible to every job, so
   the upload would need no approval.
1. Add `HF_READ_TOKEN`, a read-only Hub token for the organisation, as a repository
   secret. The `download` job, which isn't in the environment, uses it.
1. Set the repository variables `HUB_ORG` (the organisation to publish to) and
   `CI_USER_IMAGE` (a ROCm PyTorch image whose Torch and ROCm versions match a variant
   the build makes).

### Pull requests

- Actions settings: "Require approval for all external contributors". Pull request
  code runs on the MI350 runners once a run is approved; with "first-time
  contributors", a contributor whose first pull request was merged needs no approval
  after that.
- Apply [`.github/rulesets/main.json`](../.github/rulesets/main.json) to `main`.
- Make the `rocm-community-kernels-ci` package public, so the MI350 runners can pull it
  without credentials.

### GPU test runner

The runner `GPU_TEST_RUNNER` names runs pull request code, and publishing builds there
too. It needs to be:

- ephemeral: a fresh runner for every job, so nothing a pull request job leaves
  behind (work directory, Docker image cache) reaches a later job;
- limited in outbound network access to what the jobs need: GitHub and GHCR, the Nix
  caches (`cache.nixos.org`, `huggingface.cachix.org`) and PyPI.

No job on it gets a secret; `scripts/tests` checks the workflows for that, and that no
workflow names a runner label of its own.
