#pragma once

#include <torch/torch.h>

// Implemented in my_kernel_hip/scale.hip.
void scale(torch::Tensor &out, torch::Tensor const &input, double factor);
