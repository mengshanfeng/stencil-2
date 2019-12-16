#pragma once

#include "dim3.cuh"

#if STENCIL_USE_CUDA == 1
#include "cuda_runtime.hpp"
#endif

enum class storage_type {
  cpu,
#if STENCIL_USE_CUDA == 1
  device,
  managed,
#endif
};

enum class storage_order {
  xyz,
  xzy,
  yxz,
  yzx,
  zxy,
  zyx,
};

template <typename T, storage_type ST = storage_type::cpu> class Array {
private:
  Dim3 size_;
  T *data_;

public:
  Array() : size_(Dim3(0, 0, 0)), data_(nullptr) {}
  Array(Dim3 d) : Array() { resize(d); }
  ~Array() { resize(Dim3(0, 0, 0)); }

  void resize(const Dim3 &d) {
    if (d != size_) {
      delete[] data_;
      data_ = nullptr;

      size_t n = d.flatten();
      if (n > 0) {
        data_ = new T[n];
      }
      size_ = d;
    }
  }

  Dim3 size() const noexcept { return size_; }
  T *data() noexcept { return data_; }

  const T *data() const noexcept { return data_; }
  T &operator[](size_t n) { return data_[n]; }

  const T &operator[](size_t n) const { return data_[n]; }

  friend void swap(Array &a, Array &b) {
    using std::swap;
    swap(a.size_, b.size_);
    swap(a.data_, b.data_);
  }

#if STENCIL_USE_CUDA == 1
  Array<T, storage_type::device> to_gpu_async(const int dev = 0,
                                              cudaStream_t stream = 0) const;
#endif
};

#if STENCIL_USE_CUDA == 1
template <typename T> class Array<T, storage_type::device> {
private:
  Dim3 size_;
  T *data_;
  int dev_;

public:
  Array(int dev = 0) : size_(Dim3(0, 0, 0)), data_(nullptr), dev_(dev) {}
  Array(const Dim3 &d, int dev = 0) : Array(dev) { resize(d); }

  void resize(const Dim3 &d) {
    if (d != size_) {
      CUDA_RUNTIME(cudaFree(data_));
      data_ = nullptr;

      size_t n = d.flatten();
      if (n > 0) {
        CUDA_RUNTIME(cudaSetDevice(dev_));
        CUDA_RUNTIME(cudaMalloc(&data_, n * sizeof(T)));
      }
      size_ = d;
    }
  }

  Dim3 size() const noexcept { return size_; }
  T *data() noexcept { return data_; }

  const T *data() const noexcept { return data_; }
  T &operator[](size_t n) { return data_[n]; }

  const T &operator[](size_t n) const { return data_[n]; }

  friend void swap(Array &a, Array &b) {
    using std::swap;
    swap(a.size_, b.size_);
    swap(a.data_, b.data_);
    swap(a.dev_, b.dev_);
  }
};
#endif

#if STENCIL_USE_CUDA == 1
/*! Convert a CPU array to a GPU array
 */
template <typename T, storage_type ST>
Array<T, storage_type::device>
Array<T, ST>::to_gpu_async(const int dev, cudaStream_t stream) const {
  Array<T, storage_type::device> result(size(), dev);
  CUDA_RUNTIME(cudaMemcpyAsync(result.data(), data(),
                               size().flatten() * sizeof(T), cudaMemcpyDefault,
                               stream));
  return result;
};
#endif