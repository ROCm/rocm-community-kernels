{
  description = "rocm-community-kernels tooling";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.follows = "kernels/nixpkgs";
    # The kernel-builder the CI runs (check-config, check-abi, and the upload with the
    # Hub write token), pinned by flake.lock. The same tag as the kernel template.
    kernels.url = "github:huggingface/kernels/torch-2.15.0-final";
  };

  outputs =
    {
      self,
      flake-utils,
      kernels,
      nixpkgs,
    }:
    let
      systems = with flake-utils.lib.system; [
        aarch64-darwin
        aarch64-linux
        x86_64-linux
      ];
    in
    flake-utils.lib.eachSystem systems (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShell =
          with pkgs;
          mkShell {
            name = "rocm-community-kernels-dev-shell";
            nativeBuildInputs = [
              pinact
            ];
          };

        formatter = pkgs.nixfmt-tree;

        packages.kernel-builder = kernels.packages.${system}.kernel-builder;
      }
    );
}
