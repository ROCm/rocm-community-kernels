# my-kernel

> [!IMPORTANT]
> A community kernel: not supported by AMD. Report issues to its maintainers.

One or two sentences on what the kernel does and when to use it.

- **Source:** link to where the kernel originally comes from (or "written for this
  repository").
- **Licence:** the licence in `build.toml`, with its text in `LICENSE`.
- **Tested on:** the GPUs in `rocm-archs`, e.g. MI300X (gfx942) and MI350 (gfx950).
- **Maintainers:** @github-handle

## Usage

```python
import torch
from kernels import get_kernel

my_kernel = get_kernel("rocm-community/my-kernel", version=1)
y = my_kernel.scale(torch.randn(1024, device="cuda"), 2.0)
```
