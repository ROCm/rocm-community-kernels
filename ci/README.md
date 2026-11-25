# CI configuration

[`community.toml`](community.toml) holds the rules CI enforces for community kernels:
the Hub organisation, the reserved names and organisations, the GPUs CI tests on, and
the card notice. `scripts/check_wrapping.py --community` reads it, and so does the
`validate` stage of [Kernel CI](../.github/workflows/kernel-ci.yml).

The CI itself:

| Workflow                                              | When                    | What                                                                                                                                                                                                                             |
| ----------------------------------------------------- | ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`kernel-ci.yml`](../.github/workflows/kernel-ci.yml) | every pull request      | the CI scripts' unit tests, and `scripts/kernel_ci.py` for each changed kernel: validate, then build one variant and run the `kernels_ci` tests on the MI350 ([`rocm-build-test.yml`](../.github/workflows/rocm-build-test.yml)) |
| [`stage.yml`](../.github/workflows/stage.yml)         | on request, from `main` | for one pull request's kernel: build every variant, approval, upload privately to the staging organisation, then `get_kernel()` from there and test; reports "Hub staging (<kernel>)" and a comment on the pull request          |
| [`publish.yml`](../.github/workflows/publish.yml)     | on request, from `main` | build every variant and test it, approval, sign and upload, then `get_kernel()` from the Hub and test again                                                                                                                      |
| [`ci-image.yml`](../.github/workflows/ci-image.yml)   | changes to `ci/docker/` | build the CI image; from `main` (and `ci_setup` until the CI is merged into `main`), push it as `latest`, the tag the GPU jobs run                                                                                               |
| [`nix-fmt.yml`](../.github/workflows/nix-fmt.yml)     | changes to `.nix` files | `nix fmt` check                                                                                                                                                                                                                  |

Run the pull request stages locally (Nix and an AMD GPU for build and test):

```bash
python3 scripts/kernel_ci.py run --kernel <kernel> --ci
```

## Repository setup

What an administrator sets up outside the repository.

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

### Staging

Stage kernel ([`stage.yml`](../.github/workflows/stage.yml)) lets an AMD maintainer
check a pull request's kernel on the Hub before approving it: upload, download,
`get_kernel()` and the tests, in a staging organisation. Starting it needs write access
to the repository, so contributors can't, and their pull requests don't wait for it.
It checks steps 1 and 2 before it builds anything, as publishing does.

1. Create the `hf-staging` environment, with the same required reviewers and `main`
   only.
1. Add `HF_STAGING_TOKEN`, a Hub write token for the staging organisation, as a secret
   of that environment only.
1. Add `HF_STAGING_READ_TOKEN`, a read-only token for the staging organisation, as a
   repository secret.
1. Set the repository variable `HUB_STAGING_ORG` (the staging organisation). It uses
   `CI_USER_IMAGE` too.

Each pull request goes to its own branch, `pr-<number>`, of `<HUB_STAGING_ORG>/<name>`,
created private. Delete staged repositories or branches when you no longer need them.

### Pull requests

- Actions settings: "Require approval for all external contributors". Pull request
  code runs on the GPU runners once a run is approved; with "first-time
  contributors", a contributor whose first pull request was merged needs no approval
  after that.
- Apply [`.github/rulesets/main.json`](../.github/rulesets/main.json) to `main`.
- Make the `rocm-community-kernels-ci` package public, so the GPU runners can pull it
  without credentials.

### GPU runners

`[ci-runners]` in [`community.toml`](community.toml) maps each GPU CI tests on to its
runner label, for example `gfx950 = "linux-rck-mi350-1"`. A kernel is tested on each
GPU in its `rocm-archs` that is listed there, and only those, and each of them gates
its pull requests. Publishing builds and tests on the same GPUs. To add hardware, add a
line there, in a pull request, once the runner meets the requirements below. The labels
live in the repository, not in variables: pull requests from forks don't get the
repository's variables.

Each runner needs one GPU of that kind, a Docker daemon the job can use (for a
Kubernetes runner set, a mode with one, such as dind) and able to pull the CI image,
and `/dev/kfd` and `/dev/dri`, or the render devices of its own GPU listed in
`/etc/podinfo/gha-render-devices`. The jobs start the image with `docker run` and
run in it with `docker exec`; the actions run on the runner itself, so its runner
version must support the Node they use (node24). The job checks it got the GPU it was
routed for. GPU runners run pull request code, and publishing builds there too. They
need to be:

- ephemeral: a fresh runner for every job, so nothing a pull request job leaves
  behind (work directory, Docker image cache) reaches a later job;
- limited in outbound network access to what the jobs need: GitHub and GHCR, the Nix
  caches (`cache.nixos.org`, `huggingface.cachix.org`) and PyPI.

No job on them gets a secret; `scripts/tests` checks the workflows for that, and that
no workflow names a runner label of its own.
