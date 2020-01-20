#pragma once

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <set>
#include <vector>

#include <mpi.h>

#include <nvToolsExt.h>
#include <nvml.h>

#include "cuda_runtime.hpp"

#include "stencil/dim3.hpp"
#include "stencil/direction_map.hpp"
#include "stencil/gpu_topo.hpp"
#include "stencil/local_domain.cuh"
#include "stencil/nvml.hpp"
#include "stencil/partition.hpp"
#include "stencil/tx.hpp"

class DistributedDomain {
private:
  Dim3 size_;

  int rank_;
  int worldSize_;

  // the GPUs this MPI rank will use
  std::vector<int> gpus_;

  // the stencil radius
  size_t radius_;

  // typically one per GPU
  // the actual data associated with this rank
  std::vector<LocalDomain> domains_;
  // the index of the domain in the distributed domain
  std::vector<Dim3> domainIdx_;

  // information about mapping of computation domain to workers
  Partition *partition_;

  // senders/recvers for each direction in each domain
  // domainDirSenders[domainIdx].at(dir)
  // a sender/recver pair is associated with the send direction
  std::vector<DirectionMap<HaloSender *>> domainDirSender_;
  std::vector<DirectionMap<HaloRecver *>> domainDirRecver_;

  // the size in bytes of each data type
  std::vector<size_t> dataElemSize_;

  // MPI ranks co-located with me
  std::set<int64_t> colocated_;

  std::vector<std::vector<bool>> peerAccess_; //<! which GPUs have peer access

public:
  DistributedDomain(size_t x, size_t y, size_t z) : size_(x, y, z) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize_);
    int deviceCount;
    CUDA_RUNTIME(cudaGetDeviceCount(&deviceCount));

    // create a communicator for ranks on the same node
    MPI_Barrier(MPI_COMM_WORLD); // to stabilize co-located timing
    double start = MPI_Wtime();
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &shmcomm);
    int shmrank, shmsize;
    MPI_Comm_rank(shmcomm, &shmrank);
    MPI_Comm_size(shmcomm, &shmsize);
    printf("DistributedDomain::ctor(): shmcomm rank %d/%d\n", shmrank, shmsize);


    // Give every rank a list of co-located ranks
    std::vector<int> colocated(shmsize);
    MPI_Allgather(&rank_, 1, MPI_INT, colocated.data(), 1, MPI_INT, shmcomm);
    for (auto &r : colocated) {
      colocated_.insert(r);
    }
    double elapsed = MPI_Wtime() - start;
    printf("time.colocate [%d] %fs\n", rank_, elapsed);
    assert(colocated_.count(rank_) == 1 && "should be colocated with self");
    printf(
        "DistributedDomain::ctor(): rank %d colocated with %lu other ranks\n",
        rank_, colocated_.size() - 1);


    // if fewer ranks than GPUs, round-robin GPUs to ranks
    if (shmsize <= deviceCount) {
      for (int gpu = 0; gpu < deviceCount; ++gpu) {
        if (gpu % shmsize == shmrank) {
          gpus_.push_back(gpu);
        }
      }
    } else { // if more ranks, share gpus among ranks
      gpus_.push_back(shmrank % deviceCount);
    }

    for (const auto gpu : gpus_) {
      printf("rank %d/%d local=%d using gpu %d\n", rank_, worldSize_, shmrank,
             gpu);
    }


    start = MPI_Wtime();

    // Try to enable peer access between all GPUs
    nvtxRangePush("peer_en");

    // can't use gpus_.size() because we don't own all the GPUs
    peerAccess_ =
        std::vector<std::vector<bool>>(deviceCount, std::vector<bool>(deviceCount, false));

    for (int src = 0; src < deviceCount; ++src) {
      for (int dst = 0; dst < deviceCount; ++dst) {
        if (src == dst) {
          peerAccess_[src][dst] = true;
          std::cout << src << " -> " << dst << " peer access\n";
        } else {
          CUDA_RUNTIME(cudaSetDevice(src))
          cudaError_t err = cudaDeviceEnablePeerAccess(dst, 0 /*flags*/);
          if (cudaSuccess == err || cudaErrorPeerAccessAlreadyEnabled == err) {
            peerAccess_[src][dst] = true;
            std::cout << src << " -> " << dst << " peer access\n";
          } else if (cudaErrorInvalidDevice) {
            peerAccess_[src][dst] = false;
          } else {
            assert(0);
            peerAccess_[src][dst] = false;
          }
        }
      }
    }
    nvtxRangePop();
    elapsed = MPI_Wtime() - start;
    printf("time.peer [%d] %fs\n", rank_, elapsed);


    start = MPI_Wtime();
    nvtxRangePush("gpu_topo");
    Mat2D dist = get_gpu_distance_matrix();
    nvtxRangePop();
    if (0 == rank_) {
      std::cerr << "gpu distance matrix: \n";
      for (auto &r : dist) {
        for (auto &c : r) {
          std::cerr << c << " ";
        }
        std::cerr << "\n";
      }
    }
    elapsed = MPI_Wtime() - start;
    printf("time.topo [%d] %fs\n", rank_, elapsed);

    // determine decomposition information
    start = MPI_Wtime();
    nvtxRangePush("partition");
    partition_ = new PFP(size_, worldSize_, gpus_.size());
    nvtxRangePop();
    elapsed = MPI_Wtime() - start;
    printf("time.partition [%d] %fs\n", rank_, elapsed);

    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == rank_) {
      std::cerr << "split " << size_ << " into " << partition_->rank_dim()
                << "x" << partition_->gpu_dim() << "\n";
    }
  }

  ~DistributedDomain() { delete partition_; }

  std::vector<LocalDomain> &domains() { return domains_; }

  void set_radius(size_t r) { radius_ = r; }

  template <typename T> DataHandle<T> add_data() {
    dataElemSize_.push_back(sizeof(T));
    return DataHandle<T>(dataElemSize_.size() - 1);
  }

  void realize(bool useUnified = false) {

    // create local domains
    double start = MPI_Wtime();
    for (int i = 0; i < gpus_.size(); i++) {

      Dim3 idx = partition_->dom_idx(rank_, i);
      Dim3 ldSize = partition_->local_domain_size(idx);

      LocalDomain ld(ldSize, gpus_[i]);
      ld.radius_ = radius_;
      for (size_t dataIdx = 0; dataIdx < dataElemSize_.size(); ++dataIdx) {
        ld.add_data(dataElemSize_[dataIdx]);
      }

      domains_.push_back(ld);

      printf("rank=%d gpu=%d (cuda id=%d) => [%ld,%ld,%ld]\n", rank_, i,
             gpus_[i], idx.x, idx.y, idx.z);
      domainIdx_.push_back(idx);
    }

    // realize local domains
    for (auto &d : domains_) {
      if (useUnified)
        d.realize_unified();
      else
        d.realize();
      printf("DistributedDomain.realize(): finished creating LocalDomain\n");
    }
    double elapsed = MPI_Wtime() - start;
    printf("time.local_realize [%d] %fs\n", rank_, elapsed);

    // initialize null senders recvers for each domain
    domainDirSender_.resize(gpus_.size());
    domainDirRecver_.resize(gpus_.size());
    for (size_t domainIdx = 0; domainIdx < domains_.size(); ++domainIdx) {
      for (int z = 0; z < 3; ++z) {
        for (int y = 0; y < 3; ++y) {
          for (int x = 0; x < 3; ++x) {
            domainDirSender_[domainIdx].at(x, y, z) = nullptr;
            domainDirRecver_[domainIdx].at(x, y, z) = nullptr;
          }
        }
      }
    }

    const Dim3 gpuDim = partition_->gpu_dim();
    const Dim3 rankDim = partition_->rank_dim();

    // create communication plan
    start = MPI_Wtime();
    nvtxRangePush("comm plan");
    for (size_t di = 0; di < domains_.size(); ++di) {
      assert(domains_.size() == domainIdx_.size());

      auto &myDomain = domains_[di];
      const Dim3 myIdx = domainIdx_[di];
      const int myGPU = di; // logical GPU number, not device ID
      assert(rank_ == partition_->get_rank(myIdx));

      auto &dirSender = domainDirSender_[di];
      auto &dirRecver = domainDirRecver_[di];

      // send/recv pairs for faces
      for (const auto xDir : {-1, 0, 1}) {
        for (const auto yDir : {-1, 0, 1}) {
          for (const auto zDir : {-1, 0, 1}) {
            Dim3 dirVec(xDir, yDir, zDir);
            if (dirVec == Dim3(0, 0, 0)) {
              continue; // don't send in no direction
            }

            // who i am sending to for this dirVec
            Dim3 dstIdx = (myIdx + dirVec).wrap(rankDim * gpuDim);

            // who is sending to me for this dirVec
            Dim3 srcIdx = (myIdx - dirVec).wrap(rankDim * gpuDim);

            // logical GPU number, not device ID
            int srcGPU = partition_->get_gpu(srcIdx);
            int dstGPU = partition_->get_gpu(dstIdx);
            int srcRank = partition_->get_rank(srcIdx);
            int dstRank = partition_->get_rank(dstIdx);

            std::cout << myIdx << " -> " << dstIdx << " dirVec=" << dirVec
                      << " r" << rank_ << ",g" << myGPU << " -> r" << dstRank
                      << ",g" << dstGPU << "\n";

            // determine how to send face in that direction
            HaloSender *sender = nullptr;

            if (rank_ == dstRank) {
              int myCudaId = myDomain.gpu();
              int dstCudaId = domains_[dstGPU].gpu();
              if (peerAccess_[myCudaId][dstCudaId]) {
                std::cerr << "DistributedDomain.realize(): dir=" << dirVec
                          << " send same rank and peer access\n";
                sender = new RegionCopier(domains_[dstGPU], myDomain, dirVec);
              } else {
                std::cerr << "DistributedDomain.realize(): dir=" << dirVec
                          << " send same rank\n";
                sender =
                    new PackMemcpyCopier(domains_[dstGPU], myDomain, dirVec);
              }
            } else if (colocated_.count(dstRank)) { // both domains on this node
              std::cerr << "DistributedDomain.realize(): dir=" << dirVec
                        << " send colocated\n";
              sender = new RegionSender(myDomain, rank_, myGPU, dstRank, dstGPU,
                                        dirVec);
            } else { // domains on different nodes
              std::cerr << "DistributedDomain.realize(): dir=" << dirVec
                        << " send diff nodes\n";
              sender = new RegionSender(myDomain, rank_, myGPU, dstRank, dstGPU,
                                        dirVec);
            }

            std::cout << myIdx << " <- " << srcIdx << " dirVec=" << dirVec
                      << " r" << rank_ << ",g" << myGPU << " <- r" << srcRank
                      << ",g" << srcGPU << "\n";

            // determine how to receive a face from that direction
            HaloRecver *recver = nullptr;
            if (rank_ == srcRank) {
              int srcCudaId = domains_[srcGPU].gpu();
              int myCudaId = myDomain.gpu();
              if (peerAccess_[srcCudaId][myCudaId]) {
                std::cerr << "DistributedDomain.realize(): dir=" << dirVec
                          << " recv same rank and peer access\n";
                recver = nullptr; // no recver needed for RegionCopier
              } else {
                std::cerr << "DistributedDomain.realize(): dir=" << dirVec
                          << " recv same rank\n";
                recver = nullptr; // no recver needed for PackMemcpyCopier
              }
            } else if (colocated_.count(srcRank)) { // both domains on this node
              std::cerr << "DistributedDomain.realize(): dir=" << dirVec
                        << " recv colocated\n";
              recver = new RegionRecver(myDomain, srcRank, srcGPU, rank_, myGPU,
                                        dirVec);
            } else { // domains on different nodes
              std::cerr << "DistributedDomain.realize(): dir=" << dirVec
                        << " recv diff nodes\n";
              recver = new RegionRecver(myDomain, srcRank, srcGPU, rank_, myGPU,
                                        dirVec);
            }

            if (sender) {
              sender->allocate();
            }
            if (recver) {
              recver->allocate();
            }
            dirSender.at_dir(dirVec.x, dirVec.y, dirVec.z) = sender;
            dirRecver.at_dir(dirVec.x, dirVec.y, dirVec.z) = recver;
          }
        }
      }
    }
    nvtxRangePop(); // comm plan
    elapsed = MPI_Wtime() - start;
    printf("time.plan [%d] %fs\n", rank_, elapsed);
  }

  /* issue async sends for a domain
   */
  void send(const size_t domainIdx) {
    double start = MPI_Wtime();
    assert(domainIdx < domainDirSender_.size());
    auto &dirSenders = domainDirSender_[domainIdx];
    for (int z = 0; z < 3; ++z) {
      for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {
          if (auto sender = dirSenders.at(x, y, z)) {
            sender->send();
          }
        }
      }
    }
    double elapsed = MPI_Wtime() - start;
    printf("time.issue_send [%d] [%lu] %fs\n", rank_, domainIdx, elapsed);
  }

  /* issue async recvs for a domain
   */
  void recv(const size_t domainIdx) {
    double start = MPI_Wtime();
    assert(domainIdx < domainDirRecver_.size());
    auto &dirRecvers = domainDirRecver_[domainIdx];
    for (int z = 0; z < 3; ++z) {
      for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {
          if (auto recver = dirRecvers.at(x, y, z)) {
            recver->recv();
          }
        }
      }
    }
    double elapsed = MPI_Wtime() - start;
    printf("time.issue_recv [%d] [%lu] %fs\n", rank_, domainIdx, elapsed);
  }

  /*!
  do a halo exchange and return
  */
  void exchange() {
    MPI_Barrier(MPI_COMM_WORLD); // stabilize time

    double start = MPI_Wtime();

    // the sends and recvs for each domain issued async
    std::vector<std::future<void>> sends(domains_.size());
    std::vector<std::future<void>> recvs(domains_.size());

    // issue all sends
    nvtxRangePush("issue sends");
    for (size_t idx = 0; idx < domainDirSender_.size(); ++idx) {
      sends[idx] =
          std::async(std::launch::async, &DistributedDomain::send, this, idx);
    }
    nvtxRangePop(); // issue sends

    // issue all recvs
    nvtxRangePush("issue recvs");
    for (size_t idx = 0; idx < domainDirSender_.size(); ++idx) {
      recvs[idx] =
          std::async(std::launch::async, &DistributedDomain::recv, this, idx);
    }
    nvtxRangePop(); // issue recvs

    // wait for all sends and recvs to be issued
    for (size_t idx = 0; idx < domainDirSender_.size(); ++idx) {
      sends[idx].wait();
      recvs[idx].wait();
    }

    // wait for all sends and recvs
    nvtxRangePush("wait");
    for (size_t domainIdx = 0; domainIdx < domains_.size(); ++domainIdx) {
      auto &dirSender = domainDirSender_[domainIdx];
      auto &dirRecver = domainDirRecver_[domainIdx];
      for (int z = 0; z < 3; ++z) {
        for (int y = 0; y < 3; ++y) {
          for (int x = 0; x < 3; ++x) {
            if (auto recver = dirRecver.at(x, y, z)) {
              recver->wait();
            }
            if (auto sender = dirSender.at(x, y, z)) {
              sender->wait();
            }
          }
        }
      }
    }
    nvtxRangePop(); // wait

    double elapsed = MPI_Wtime() - start;
    printf("time.exchange [%d] %fs\n", rank_, elapsed);

    // wait for all ranks to be done
    MPI_Barrier(MPI_COMM_WORLD);
  }
};
