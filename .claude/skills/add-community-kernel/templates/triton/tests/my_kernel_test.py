import kernels
import pytest
import torch

# The repo id and version from build.toml. In CI, LOCAL_KERNELS points this at the
# fresh build; never import the package directly.
my_kernel = kernels.get_kernel("rocm-community/my-kernel", version=1)


# The kernels_ci subset runs on the GPU in CI: keep it under a minute in total.
@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_scale(dtype):
    x = torch.randn(10_000, device="cuda", dtype=dtype)
    torch.testing.assert_close(my_kernel.scale(x, 2.0), x * 2.0)


@pytest.mark.kernels_ci
def test_scale_empty():
    x = torch.empty(0, device="cuda")
    assert my_kernel.scale(x, 2.0).numel() == 0
