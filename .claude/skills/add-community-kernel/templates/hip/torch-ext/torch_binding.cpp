#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

// Register under the build-generated namespace, never a fixed one: two kernels
// (or two versions of one kernel) must be able to load side by side.
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("scale(Tensor! out, Tensor input, float factor) -> ()");
  // ROCm builds of PyTorch use the CUDA dispatch key.
  ops.impl("scale", torch::kCUDA, &scale);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
