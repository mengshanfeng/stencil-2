#pragma once

#include "tx_common.hpp"

#if STENCIL_USE_CUDA == 1 && __NVCC__
#include "tx_cuda.cuh"
#endif