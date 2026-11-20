import kernels
import pytest
import torch

# The repo id and version from build.toml. In CI, LOCAL_KERNELS points this at the
# fresh build; never import the package directly.
my_kernel = kernels.get_kernel("rocm-community/my-kernel", version=1)


# The kernels_ci subset runs on the GPU in CI: keep it under a minute in total.
@pytest.mark.kernels_ci
@pytest.mark.parametrize("n", [1, 255, 256, 10_000])
def test_scale(n):
    x = torch.randn(n, device="cuda")
    torch.testing.assert_close(my_kernel.scale(x, 2.0), x * 2.0)


@pytest.mark.kernels_ci
def test_scale_into_out():
    x = torch.randn(1024, device="cuda")
    out = torch.empty_like(x)
    assert my_kernel.scale(x, 3.0, out=out) is out
    torch.testing.assert_close(out, x * 3.0)
