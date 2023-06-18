#ifndef __MPI_CONFIG_HPP__
#define __MPI_CONFIG_HPP__

#include <cassert>
#include <mpi.h>

struct MPIConfig {
private:
  // ID of the MPI process
  int rank_; 

  // Number of MPI processes
  int size_;

  // Communicator
  MPI_Comm communicator_;

  bool is_initialized;

public:
  MPIConfig() : is_initialized(false) {}
  ~MPIConfig() {}

public:
  void initialize(int* argc, char*** argv) {
    is_initialized = true;
    communicator_ = MPI_COMM_WORLD;
    int required = MPI_THREAD_MULTIPLE;
    int provided;
    ::MPI_Init_thread(argc, argv, required, &provided);
    ::MPI_Comm_size(MPI_COMM_WORLD, &size_);
    ::MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  }

  void finalize() { if(is_initialized) ::MPI_Finalize(); }
  bool is_master() { return rank_==0; }
  int size() const { return size_; }
  int rank() const { return rank_; }
  auto comm() const { return communicator_; }
  void fence() const { MPI_Barrier(communicator_); }
};

#endif
