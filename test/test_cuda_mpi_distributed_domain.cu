#include "catch2/catch.hpp"

#include <cstring> // std::memcpy

#include "stencil/copy.cuh"
#include "stencil/cuda_runtime.hpp"
#include "stencil/dim3.hpp"
#include "stencil/stencil.hpp"

__device__ int pack_xyz(int x, int y, int z) {
  int ret = 0;
  ret |= x & 0x3FF;
  ret |= (y & 0x3FF) << 10;
  ret |= (z & 0x3FF) << 20;
  return ret;
}

int unpack_x(int a) { return a & 0x3FF; }

int unpack_y(int a) { return (a >> 10) & 0x3FF; }

int unpack_z(int a) { return (a >> 20) & 0x3FF; }

/*! set dst[x,y,z] = origin + [x,y,z] in interior
and halo to -1
*/
template <typename T>
__global__ void
init_kernel(T *dst,            //<! [out] pointer to beginning of dst allocation
            const Dim3 origin, //<! [in]
            const Dim3 rawSz   //<! [in] 3D size of the dst and src allocations
) {
  constexpr size_t radius = 1;
  const Dim3 domSz = rawSz - Dim3(2 * radius, 2 * radius, 2 * radius);

  const size_t gdz = gridDim.z;
  const size_t biz = blockIdx.z;
  const size_t bdz = blockDim.z;
  const size_t tiz = threadIdx.z;

  const size_t gdy = gridDim.y;
  const size_t biy = blockIdx.y;
  const size_t bdy = blockDim.y;
  const size_t tiy = threadIdx.y;

  const size_t gdx = gridDim.x;
  const size_t bix = blockIdx.x;
  const size_t bdx = blockDim.x;
  const size_t tix = threadIdx.x;

#ifndef _at
#define _at(arr, _x, _y, _z) arr[_z * rawSz.y * rawSz.x + _y * rawSz.x + _x]
#else
#error "_at already defined"
#endif

  for (size_t z = biz * bdz + tiz; z < rawSz.z; z += gdz * bdz) {
    for (size_t y = biy * bdy + tiy; y < rawSz.y; y += gdy * bdy) {
      for (size_t x = bix * bdx + tix; x < rawSz.x; x += gdx * bdx) {

        if (z >= radius && x >= radius && y >= radius && z < rawSz.z - radius &&
            y < rawSz.y - radius && x < rawSz.x - radius) {
          _at(dst, x, y, z) =
              pack_xyz(origin.x + x - radius, origin.y + y - radius,
                       origin.z + z - radius);
        } else {
          _at(dst, x, y, z) = -1;
        }
      }
    }
  }

#undef _at
}

TEST_CASE("exchange1") {

  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  srand(time(NULL) + rank);

  size_t radius = 1;
  typedef float Q1;

  INFO("ctor");
  DistributedDomain dd(10, 10, 10);
  dd.set_radius(radius);
  auto dh1 = dd.add_data<Q1>("d0");
  dd.set_methods(MethodFlags::CudaMpi);

  INFO("realize");
  dd.realize();

  INFO("device sync");
  for (auto &d : dd.domains()) {
    CUDA_RUNTIME(cudaSetDevice(d.gpu()));
    CUDA_RUNTIME(cudaDeviceSynchronize());
  }

  INFO("barrier");
  MPI_Barrier(MPI_COMM_WORLD);

  INFO("init");
  dim3 dimGrid(10, 10, 10);
  dim3 dimBlock(8, 8, 8);
  for (size_t di = 0; di < dd.domains().size(); ++di) {
    auto &d = dd.domains()[di];
    REQUIRE(d.get_curr(dh1) != nullptr);
    std::cerr << d.raw_size() << "\n";
    CUDA_RUNTIME(cudaSetDevice(d.gpu()));
    init_kernel<<<dimGrid, dimBlock>>>(d.get_curr(dh1), d.origin(),
                                       d.raw_size());
    CUDA_RUNTIME(cudaDeviceSynchronize());
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // test initialization
  INFO("test init interior");
  for (size_t di = 0; di < dd.domains().size(); ++di) {
    auto &d = dd.domains()[di];
    const Dim3 origin = d.origin();
    const Dim3 ext = d.halo_extent(Dim3(0, 0, 0));

    for (size_t qi = 0; qi < d.num_data(); ++qi) {
      auto vec = d.interior_to_host(qi);

      // make sure we can access data as a Q1
      std::vector<Q1> interior(ext.flatten());
      REQUIRE(vec.size() == interior.size() * sizeof(Q1));
      std::memcpy(interior.data(), vec.data(), vec.size());

      for (int64_t z = 0; z < ext.z; ++z) {
        for (int64_t y = 0; y < ext.y; ++y) {
          for (int64_t x = 0; x < ext.x; ++x) {
            Q1 val = interior[z * (ext.y * ext.x) + y * (ext.x) + x];
            REQUIRE(unpack_x(val) == x + origin.x);
            REQUIRE(unpack_y(val) == y + origin.y);
            REQUIRE(unpack_z(val) == z + origin.z);
          }
        }
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  INFO("exchange");

  dd.exchange();
  CUDA_RUNTIME(cudaDeviceSynchronize());

  INFO("interior should be unchanged");
  for (size_t di = 0; di < dd.domains().size(); ++di) {
    auto &d = dd.domains()[di];
    const Dim3 origin = d.origin();
    const Dim3 ext = d.halo_extent(Dim3(0, 0, 0));

    for (size_t qi = 0; qi < d.num_data(); ++qi) {
      auto vec = d.interior_to_host(qi);

      // make sure we can access data as a Q1
      std::vector<Q1> interior(ext.flatten());
      REQUIRE(vec.size() == interior.size() * sizeof(Q1));
      std::memcpy(interior.data(), vec.data(), vec.size());

      for (int64_t z = 0; z < ext.z; ++z) {
        for (int64_t y = 0; y < ext.y; ++y) {
          for (int64_t x = 0; x < ext.x; ++x) {
            Q1 val = interior[z * (ext.y * ext.x) + y * (ext.x) + x];
            REQUIRE(unpack_x(val) == x + origin.x);
            REQUIRE(unpack_y(val) == y + origin.y);
            REQUIRE(unpack_z(val) == z + origin.z);
          }
        }
      }
    }
  }

  INFO("check halo regions");

  for (size_t di = 0; di < dd.domains().size(); ++di) {
    auto &d = dd.domains()[di];
    const Dim3 origin = d.origin();

    Dim3 ext = d.size();
    ext.x += 2 * radius;
    ext.y += 2 * radius;
    ext.z += 2 * radius;

    for (size_t qi = 0; qi < d.num_data(); ++qi) {
      auto vec = d.quantity_to_host(qi);
      // access quantity data as a Q1
      std::vector<Q1> quantity(ext.flatten());
      REQUIRE(vec.size() == quantity.size() * sizeof(Q1));
      std::memcpy(quantity.data(), vec.data(), vec.size());

      for (int64_t z = 0; z < ext.z; ++z) {
        for (int64_t y = 0; y < ext.y; ++y) {
          for (int64_t x = 0; x < ext.x; ++x) {
            Dim3 xyz = Dim3(x, y, z);
            Dim3 coord = xyz - Dim3(radius, radius, radius) + origin;
            coord = coord.wrap(Dim3(10, 10, 10));

            Q1 val = quantity[z * (ext.y * ext.x) + y * (ext.x) + x];
            REQUIRE(unpack_x(val) == coord.x);
            REQUIRE(unpack_y(val) == coord.y);
            REQUIRE(unpack_z(val) == coord.z);
          }
        }
      }
    }
  }
}


TEST_CASE("swap") {
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  srand(time(NULL) + rank);

  size_t radius = 1;
  typedef float Q1;

  INFO("ctor");
  DistributedDomain dd(10, 10, 10);
  dd.set_radius(radius);
  auto dh1 = dd.add_data<Q1>("d0");
  dd.set_methods(MethodFlags::CudaMpi);

  INFO("realize");
  dd.realize();

  INFO("device sync");
  for (auto &d : dd.domains()) {
    CUDA_RUNTIME(cudaSetDevice(d.gpu()));
    CUDA_RUNTIME(cudaDeviceSynchronize());
  }

  INFO("barrier");
  MPI_Barrier(MPI_COMM_WORLD);

  dd.swap();
}