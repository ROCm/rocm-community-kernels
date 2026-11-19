import torch
import triton
import triton.language as tl


@triton.jit
def _scale_kernel(x_ptr, out_ptr, n, factor, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * factor, mask=mask)


def scale(x: torch.Tensor, factor: float) -> torch.Tensor:
    """Return x * factor, computed by a Triton kernel."""
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    if n:
        grid = (triton.cdiv(n, 1024),)
        _scale_kernel[grid](x, out, n, factor, BLOCK=1024)
    return out
