# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The CI image: what the gfx950 "Build and GPU test" job runs in
# (.github/workflows/rocm-build-test.yml). Slim on purpose: Python, git and Nix
# using Hugging Face's binary cache. No ROCm and no PyTorch: Nix fetches the exact
# ones each kernel variant is built and tested with.
#
# Built and pushed to the GitHub Container Registry by .github/workflows/ci-image.yml.
# It must never contain a secret: anyone can pull it, and pull request code runs in it.

FROM ubuntu:24.04@sha256:73b999393d07f2bf65a40373778a44dc9a6d54ad7718d29247d8ff9413c4d6a8

ARG NIX_INSTALLER_VERSION=v3.24.2

# Python 3.12 (as python and python3) runs scripts/kernel_ci.py; git is for
# actions/checkout and Nix flakes; xz-utils for the Nix installer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git xz-utils python3 python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

# Determinate Nix, pinned. --init none: no daemon (a container has no init system).
# sandbox = false: Docker's default seccomp profile blocks Nix's build sandbox.
# Hugging Face's binary cache holds the prebuilt PyTorch x ROCm toolchains;
# kernel_ci.py fails the build without it.
RUN curl --proto '=https' --tlsv1.2 -fsSL \
        "https://install.determinate.systems/nix/tag/${NIX_INSTALLER_VERSION}" \
    | sh -s -- install linux --init none --no-confirm \
        --extra-conf "sandbox = false" \
        --extra-conf "extra-substituters = https://huggingface.cachix.org" \
        --extra-conf "extra-trusted-public-keys = huggingface.cachix.org-1:ynTPbLS0W8ofXd9fDjk1KvoFky9K2jhxe6r4nXAkc/o="

ENV PATH=/nix/var/nix/profiles/default/bin:$PATH \
    NIX_REMOTE=local

# Fail the image build if Nix doesn't work or doesn't use Hugging Face's cache.
RUN nix --version \
    && nix config show | grep -q 'huggingface.cachix.org' \
    && python --version

# The commit the image was built from (ci-image.yml passes it); the GPU job prints it.
ARG REVISION=unknown
ENV CI_IMAGE_REVISION=$REVISION
