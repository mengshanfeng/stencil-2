#pragma once

#include "dim3.hpp"
#include "pack_kernel.cuh"

// pitch calculations
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g32bd7a39135594788a542ae72217775c



/*! same as calling pack(&dst[offsets[i]]...srcs[i]...elemSizes[i])
 */
static __global__ void
multi_pack(void *__restrict__ dst,                      // dst buffer
           const size_t *__restrict__ offsets,          // offsets into dst
           void *__restrict__ *__restrict__ const srcs, // n src pointers
           const Dim3 srcSize, const Dim3 srcPos, const Dim3 srcExtent,
           const size_t *__restrict__ elemSizes, // n elem sizes
           const size_t n) {
  for (size_t i = 0; i < n; ++i) {
    void *dstp = &(static_cast<char *>(dst)[offsets[i]]);
    grid_pack(dstp, srcs[i], srcSize, srcPos, srcExtent, elemSizes[i]);
  }
}

inline __device__ void grid_unpack(void *__restrict__ dst, const Dim3 dstSize,
                                   const Dim3 dstPos, const Dim3 dstExtent,
                                   const void *__restrict__ src,
                                   const size_t elemSize) {

  const size_t tz = blockDim.z * blockIdx.z + threadIdx.z;
  const size_t ty = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t tx = blockDim.x * blockIdx.x + threadIdx.x;

  for (size_t zi = tz; zi < dstExtent.z; zi += blockDim.z * gridDim.z) {
    for (size_t yi = ty; yi < dstExtent.y; yi += blockDim.y * gridDim.y) {
      for (size_t xi = tx; xi < dstExtent.x; xi += blockDim.x * gridDim.x) {
        size_t zo = zi + dstPos.z;
        size_t yo = yi + dstPos.y;
        size_t xo = xi + dstPos.x;
        size_t oi = zo * dstSize.y * dstSize.x + yo * dstSize.x + xo;
        size_t ii = zi * dstExtent.y * dstExtent.x + yi * dstExtent.x + xi;
        // printf("%lu %lu %lu [%lu] -> %lu %lu %lu [%lu]\n", xi, yi, zi, ii,
        // xo,
        //        yo, zo, oi);
        if (4 == elemSize) {
          uint32_t *pDst = reinterpret_cast<uint32_t *>(dst);
          const uint32_t *pSrc = reinterpret_cast<const uint32_t *>(src);
          pDst[oi] = pSrc[ii];
        } else if (8 == elemSize) {
          uint64_t *pDst = reinterpret_cast<uint64_t *>(dst);
          const uint64_t *pSrc = reinterpret_cast<const uint64_t *>(src);
          pDst[oi] = pSrc[ii];
        } else {
          char *pDst = reinterpret_cast<char *>(dst);
          const char *pSrc = reinterpret_cast<const char *>(src);
          memcpy(&pDst[oi * elemSize], &pSrc[ii * elemSize], elemSize);
        }
      }
    }
  }
}

static __global__ void unpack(void *__restrict__ dst, const Dim3 dstSize,
                              const size_t dstPitch, const Dim3 dstPos,
                              const Dim3 dstExtent,
                              const void *__restrict__ src,
                              const size_t elemSize) {

  grid_unpack(dst, dstSize, dstPos, dstExtent, src, elemSize);
}

/*! same as calling unpack(dsts[i]...srcs[offsets[i]]...elemSizes[i])
 */
static __global__ void
multi_unpack(void **__restrict__ dsts, const Dim3 dstSize, const Dim3 dstPos, const Dim3 dstExtent,
             const void *__restrict__ const src, const size_t *__restrict__ offsets, 
             const size_t *__restrict__ elemSizes, const size_t n) {
  for (size_t i = 0; i < n; ++i) {
    const void *srcp = &(static_cast<const char *>(src)[offsets[i]]);
    grid_unpack(dsts[i], dstSize, dstPos, dstExtent, srcp, elemSizes[i]);
  }
}

inline __device__ void
translate_grid(void *__restrict__ dst, const Dim3 dstPos, const Dim3 dstSize,
               const void *__restrict__ src, const Dim3 srcPos,
               const Dim3 srcSize,
               const Dim3 extent, // the extent of the region to be copied
               const size_t elemSize) {

  char *cDst = reinterpret_cast<char *>(dst);
  const char *cSrc = reinterpret_cast<const char *>(src);

  const size_t tz = blockDim.z * blockIdx.z + threadIdx.z;
  const size_t ty = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t tx = blockDim.x * blockIdx.x + threadIdx.x;

  const Dim3 dstStop = dstPos + extent;

  for (size_t z = tz; z < extent.z; z += blockDim.z * gridDim.z) {
    for (size_t y = ty; y < extent.y; y += blockDim.y * gridDim.y) {
      for (size_t x = tx; x < extent.x; x += blockDim.x * gridDim.x) {
        // input coorindates
        size_t zi = z + srcPos.z;
        size_t yi = y + srcPos.y;
        size_t xi = x + srcPos.x;
        // output coordinates
        size_t zo = z + dstPos.z;
        size_t yo = y + dstPos.y;
        size_t xo = x + dstPos.x;
        // linearized
        size_t lo = zo * dstSize.y * dstSize.x + yo * dstSize.x + xo;
        size_t li = zi * srcSize.y * srcSize.x + yi * srcSize.x + xi;
        // printf("%lu %lu %lu [%lu] -> %lu %lu %lu [%lu]\n", xi, yi, zi, ii,
        // xo,
        //        yo, zo, oi);
        memcpy(&cDst[lo * elemSize], &cSrc[li * elemSize], elemSize);
      }
    }
  }
}

// take the 3D region src[srcPos...srcPos+extent] and translate it to the 3D
// region dst[dstPos...dstPos+extent]
// srcSize and dstSize may not be the same
static __global__ void
translate(void *__restrict__ dst, const Dim3 dstPos, const Dim3 dstSize,
          const void *__restrict__ src, const Dim3 srcPos, const Dim3 srcSize,
          const Dim3 extent, // the extent of the region to be copied
          const size_t elemSize) {

  translate_grid(dst, dstPos, dstSize, src, srcPos, srcSize, extent, elemSize);
}

static __global__ void
multi_translate(void *__restrict__ *__restrict__ dsts, const Dim3 dstPos,
                const Dim3 dstSize, void *__restrict__ *__restrict__ const srcs,
                const Dim3 srcPos, const Dim3 srcSize,
                const Dim3 extent, // the extent of the region to be copied
                size_t *const __restrict__ elemSizes, const size_t n) {
  for (size_t i = 0; i < n; ++i) {
    translate_grid(dsts[i], dstPos, dstSize, srcs[i], srcPos, srcSize, extent,
                   elemSizes[i]);
  }
}
