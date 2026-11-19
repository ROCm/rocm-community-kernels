#include <ATen/hip/HIPContext.h>
#include <c10/core/DeviceGuard.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

namespace {

__global__ void scale_kernel(float *out, const float *in, float factor,
                             int64_t n) {
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i < n) {
    out[i] = in[i] * factor;
  }
}

} // namespace

void scale(torch::Tensor &out, torch::Tensor const &input, double factor) {
  TORCH_CHECK(input.is_cuda() && out.is_cuda(), "tensors must be on the GPU");
  TORCH_CHECK(input.scalar_type() == at::kFloat &&
                  out.scalar_type() == at::kFloat,
              "only float32 is supported");
  TORCH_CHECK(input.is_contiguous() && out.is_contiguous(),
              "tensors must be contiguous");
  TORCH_CHECK(out.numel() == input.numel(), "out and input sizes differ");

  const int64_t n = input.numel();
  if (n == 0) {
    return;
  }
  const c10::DeviceGuard guard(input.device());
  const hipStream_t stream = at::hip::getCurrentHIPStream();
  constexpr int threads = 256;
  const int64_t blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(scale_kernel, dim3(blocks), dim3(threads), 0, stream,
                     out.data_ptr<float>(), input.data_ptr<float>(),
                     static_cast<float>(factor), n);
  TORCH_CHECK(hipGetLastError() == hipSuccess, "scale_kernel launch failed");
}
