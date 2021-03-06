#pragma once

#include <iostream>
#include <thread>
#include <vector>

#include "stencil/accessor.hpp"
#include "stencil/cuda_runtime.hpp"
#include "stencil/dim3.hpp"
#include "stencil/logging.hpp"
#include "stencil/pack_kernel.cuh"
#include "stencil/radius.hpp"
#include "stencil/rect3.hpp"

class DistributedDomain;

template <typename T> class DataHandle {
  friend class DistributedDomain;
  friend class LocalDomain;
  size_t id_;
  std::string name_;

public:
  DataHandle(size_t i, const std::string &name = "") : id_(i), name_(name) {}
};

enum class DataType {
  None,
  Float,
  Double,
};

class LocalDomain {
  friend class DistributedDomain;

private:
  // my local data size (not including halo)
  Dim3 sz_;

  /* coordinate of the origin of the compute region
  not necessarily the coordinate of the first allocation element
  since the halo is usually non-zero*/
  Dim3 origin_;

  //!< radius of stencils that will be applied
  Radius radius_;

  //!< backing info for the actual data I have
  // host versions
  std::vector<void *> currDataPtrs_;
  std::vector<void *> nextDataPtrs_;
  std::vector<int64_t> dataElemSize_;
  std::vector<std::string> dataName_;
  /* device versions of the pointers (the pointers already point to device data)
   used in the packers
   */
  void **devCurrDataPtrs_;
  size_t *devDataElemSize_;

  int dev_; // CUDA device

public:
  LocalDomain(Dim3 sz, Dim3 origin, int dev)
      : sz_(sz), origin_(origin), dev_(dev), devCurrDataPtrs_(nullptr), devDataElemSize_(nullptr) {}

  ~LocalDomain() {
    CUDA_RUNTIME(cudaGetLastError());

    // int rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // std::cerr << "dtor rank=" << rank << " ~LocalDomain(): device=" << dev_
    // << "\n";
    CUDA_RUNTIME(cudaSetDevice(dev_));
    for (auto p : currDataPtrs_) {
      // std::cerr << "rank=" << rank << " ~LocalDomain(): cudaFree " <<
      // uintptr_t(p) << "\n";
      if (p)
        CUDA_RUNTIME(cudaFree(p));
    }
    if (devCurrDataPtrs_)
      CUDA_RUNTIME(cudaFree(devCurrDataPtrs_));

    for (auto p : nextDataPtrs_) {
      if (p)
        CUDA_RUNTIME(cudaFree(p));
    }
    if (devDataElemSize_)
      CUDA_RUNTIME(cudaFree(devDataElemSize_));
    CUDA_RUNTIME(cudaGetLastError());
  }

  /* set the CUDA device for this LocalDomain
   */
  void set_device(CudaErrorsFatal fatal = CudaErrorsFatal::YES);

  /*
   */
  int64_t num_data() const {
    assert(currDataPtrs_.size() == nextDataPtrs_.size());
    assert(dataElemSize_.size() == currDataPtrs_.size());
    return int64_t(currDataPtrs_.size());
  }

  const Dim3 &origin() const noexcept { return origin_; }

  /*! Add an untyped data field with an element size of n.

  \returns The index of the added data
  */
  int64_t add_data(size_t n, const std::string &name = "") {
    dataName_.push_back(name);
    dataElemSize_.push_back(n);
    currDataPtrs_.push_back(nullptr);
    nextDataPtrs_.push_back(nullptr);
    return int64_t(dataElemSize_.size()) - 1;
  }

  template <typename T> DataHandle<T> add_data(const std::string &name = "") {
    return DataHandle<T>(add_data(sizeof(T)), name);
  }

  /*! \brief set the radius. Should only be called by DistributedDomain
  TODO friend class
   */
  void set_radius(size_t r) { radius_ = Radius::constant(r); }
  void set_radius(const Radius &r) {
    LOG_SPEW("in " << __FUNCTION__);
    radius_ = r;
  }

  const Radius &radius() const noexcept { return radius_; }

  /*! \brief retrieve a pointer to current domain values (to read in stencil)
   */
  template <typename T> T *get_curr(const DataHandle<T> handle) const {
    assert(dataElemSize_.size() > handle.id_);
    assert(currDataPtrs_.size() > handle.id_);
    void *ptr = currDataPtrs_[handle.id_];
    assert(sizeof(T) == dataElemSize_[handle.id_]);
    return static_cast<T *>(ptr);
  }

  /*! \brief retrieve a pointer to next domain values (to set in stencil)
   */
  template <typename T> T *get_next(const DataHandle<T> handle) const {
    assert(dataElemSize_.size() > handle.id_);
    assert(nextDataPtrs_.size() > handle.id_);
    void *ptr = nextDataPtrs_[handle.id_];
    assert(sizeof(T) == dataElemSize_[handle.id_]);
    return static_cast<T *>(ptr);
  }

  size_t elem_size(const size_t idx) const {
    assert(idx < dataElemSize_.size());
    return dataElemSize_[idx];
  }

  size_t *dev_elem_sizes() const { return devDataElemSize_; }

  void *curr_data(size_t idx) const {
    assert(idx < currDataPtrs_.size());
    return currDataPtrs_[idx];
  }

  void **dev_curr_datas() const { return devCurrDataPtrs_; }

  void *next_data(size_t idx) const {
    assert(idx < nextDataPtrs_.size());
    return nextDataPtrs_[idx];
  }

  template <typename T> Accessor<T> get_curr_accessor(const DataHandle<T> &dh) const noexcept {
    T *raw = get_curr(dh);

    // the origin stored in the localdomain does not include the halo,
    // but the accessor needs to know how to skip the halo region
    Dim3 org = origin();
    org.x -= radius_.x(-1);
    org.y -= radius_.y(-1);
    org.z -= radius_.z(-1);

    // the total allocation size, for indexing in the accessor
    Dim3 pitch = sz_;
    pitch.x += radius_.x(-1) + radius_.x(1);
    pitch.y += radius_.y(-1) + radius_.y(1);
    pitch.z += radius_.z(-1) + radius_.z(1);

    return Accessor<T>(raw, org, pitch);
  }

  template <typename T> Accessor<T> get_next_accessor(const DataHandle<T> &dh) const noexcept {
    T *raw = get_next(dh);

    // the origin stored in the localdomain does not include the halo,
    // but the accessor needs to know how to skip the halo region
    Dim3 org = origin();
    org.x -= radius_.x(-1);
    org.y -= radius_.y(-1);
    org.z -= radius_.z(-1);

    // the total allocation size, for indexing in the accessor
    Dim3 pitch = sz_;
    pitch.x += radius_.x(-1) + radius_.x(1);
    pitch.y += radius_.y(-1) + radius_.y(1);
    pitch.z += radius_.z(-1) + radius_.z(1);

    return Accessor<T>(raw, org, pitch);
  }

  /* return the coordinates of the compute region (not including the halo)
   */
  Rect3 get_compute_region() const noexcept;

  /* return the coordinates of the whole domain, including the halo
   */
  Rect3 get_full_region() const noexcept {
    Dim3 lo = origin();
    Dim3 hi = origin() + size();
    lo.x -= radius_.dir(-1, 0, 0);
    lo.y -= radius_.dir(0, -1, 0);
    lo.z -= radius_.dir(0, 0, -1);
    hi.x += radius_.dir(1, 0, 0);
    hi.y += radius_.dir(0, 1, 0);
    hi.z += radius_.dir(0, 0, 1);
    return Rect3(lo, hi);
  }

#if 0
  // return the position of the halo relative to get_data() in direction `dir`
  Dim3 halo_pos(const Dim3 &dir, const bool halo) const noexcept {
    Dim3 ret;
    assert(dir.all_gt(-2));
    assert(dir.all_lt(2));

    // +xhalo is the left edge + -x radius + the interior
    // +x interior is just the left edge + interior size
    if (1 == dir.x) {
      ret.x = sz_.x + (halo ? radius_.x(-1) : 0);
    } else if (-1 == dir.x) {
      ret.x = halo ? 0 : radius_.x(-1);
    } else if (0 == dir.x) {
      ret.x = radius_.x(-1);
    } else {
      __builtin_unreachable();
    }

    if (1 == dir.y) {
      ret.y = sz_.y + (halo ? radius_.y(-1) : 0);
    } else if (-1 == dir.y) {
      ret.y = halo ? 0 : radius_.y(-1);
    } else if (0 == dir.y) {
      ret.y = radius_.y(-1);
    } else {
      __builtin_unreachable();
    }

    if (1 == dir.z) {
      ret.z = sz_.z + (halo ? radius_.z(-1) : 0);
    } else if (-1 == dir.z) {
      ret.z = halo ? 0 : radius_.z(-1);
    } else if (0 == dir.z) {
      ret.z = radius_.z(-1);
    } else {
      __builtin_unreachable();
    }

    return ret;
  }
#endif

  // return the position of the halo relative to get_data() on the `dir` side of the
  // LocalDomain (e.g., dir [1,0,0] returns the position of the region on the +x side)
  // dir = [0,0,0] returns the entire region (without the halo)
  Dim3 halo_pos(const Dim3 &dir, const bool halo) const noexcept;

  /* return the coordinates of the halo region on side `dir`.
     `halo` determines whether exterior (false) or halo (true) coordinates
  */
  Rect3 halo_coords(const Dim3 &dir, const bool halo) const;

  /* get the point-size of the halo region on side `dir`, with a compute region of size `sz` and a kernel radius
  `radius`. dir=[0,0,0] returns sz
  */
  static Dim3 halo_extent(const Dim3 &dir, const Dim3 &sz, const Radius &radius) {
    assert(dir.x >= -1 && dir.x <= 1);
    assert(dir.y >= -1 && dir.y <= 1);
    assert(dir.z >= -1 && dir.z <= 1);
    Dim3 ret;

    ret.x = (0 == dir.x) ? sz.x : radius.x(dir.x);
    ret.y = (0 == dir.y) ? sz.y : radius.y(dir.y);
    ret.z = (0 == dir.z) ? sz.z : radius.z(dir.z);
    return ret;
  }

  // return the extent of the halo in direction `dir`
  Dim3 halo_extent(const Dim3 &dir) const noexcept { return halo_extent(dir, sz_, radius_); }

  // return the number of bytes of the halo in direction `dir`
  int64_t halo_bytes(const Dim3 &dir, const int64_t idx) const noexcept {
    return dataElemSize_[idx] * halo_extent(dir).flatten();
  }

  // return the 3d size of the compute domain, in terms of elements
  Dim3 size() const noexcept { return sz_; }

  // return the 3d size of the actual allocation, in terms of elements
  Dim3 raw_size() const noexcept {
    return Dim3(sz_.x + radius_.x(-1) + radius_.x(1), sz_.y + radius_.y(-1) + radius_.y(1),
                sz_.z + radius_.z(-1) + radius_.z(1));
  }

  // the GPU this domain is on
  int gpu() const { return dev_; }

  /* Swap current and next pointers
   */
  void swap() noexcept;

  /* return the bytes making up
  */
  std::vector<unsigned char> region_to_host(const Dim3 &pos, const Dim3 &ext,
                                            const size_t qi // quantity index
                                            ) const;

  /*! Copy the compute region to the host
   */
  std::vector<unsigned char> interior_to_host(const size_t qi // quantity index
                                              ) const {

    const Dim3 pos = halo_pos(Dim3(0, 0, 0), true);
    const Dim3 ext = halo_extent(Dim3(0, 0, 0));
    return region_to_host(pos, ext, qi);
  }

  /*! Copy an entire quantity, including halo region, to host
   */
  std::vector<unsigned char> quantity_to_host(const size_t qi // quantity index
                                              ) const {
    Dim3 allocSz = sz_;
    allocSz.x += radius_.x(-1) + radius_.x(1);
    allocSz.y += radius_.y(-1) + radius_.y(1);
    allocSz.z += radius_.z(-1) + radius_.z(1);
    return region_to_host(Dim3(0, 0, 0), allocSz, qi);
  }

  void realize();
};
