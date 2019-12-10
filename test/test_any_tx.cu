#include "catch2/catch.hpp"

#include <mpi.h>

#include "stencil/tx.cuh"
#include "stencil/cuda_runtime.hpp"

TEMPLATE_TEST_CASE("any tx", "[mpi][cuda][template]", int32_t, int64_t) {

  const size_t n = 100;

  TestType *buf0 = nullptr;
  TestType *buf1 = nullptr;

  INFO("allocate bufs");
  CUDA_RUNTIME(cudaMallocManaged(&buf0, n * sizeof(TestType)));
  CUDA_RUNTIME(cudaMallocManaged(&buf1, n * sizeof(TestType)));

  INFO("init bufs");
  for (size_t i = 0; i < n; ++i) {
      buf0[i] = i+1;
      buf1[i] = 0;
  }
  REQUIRE(buf1[0] != buf0[0]);
  REQUIRE(buf1[n-1] != buf0[n-1]);

  INFO("tx ctors");
  AnySender sender(0,0,0,0);
  AnyRecver recver(0,0,0,0);
  INFO("tx resize");
  sender.resize(n * sizeof(TestType));
  recver.resize(n * sizeof(TestType));

  INFO("send/recv");
  sender.send(buf0);
  recver.recv(buf1);

  INFO("wait");
  sender.wait();
  recver.wait();

  INFO("cuda sync");
  CUDA_RUNTIME(cudaDeviceSynchronize());

  REQUIRE(buf1[0] == buf0[0]);
  REQUIRE(buf1[n-1] == buf0[n-1]);


  CUDA_RUNTIME(cudaFree(buf0));
  CUDA_RUNTIME(cudaFree(buf1));
}