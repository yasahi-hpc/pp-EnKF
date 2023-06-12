/* Test results
   Process2 running on GPU2
   Process0 running on GPU0
   Process1 running on GPU1
   Process3 running on GPU3

   comm 0.791734 [s], 1 calls
   commh2h 8.28193 [s], 1 calls
   transpose 0.532774 [s], 1 calls
   axpy 0.106926 [s], 1 calls
   sync_comm_then 1.43611 [s], 1 calls
   Async_comm_then 1.43732 [s], 1 calls
   Async_commh2h_then 9.18571 [s], 1 calls
   Sync_comm_bulk 0.953238 [s], 1 calls
   Async_comm_bulk 0.920117 [s], 1 calls
   Dummy 25.5671 [s], 9 calls
 */

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <array>
#include <stdexec/execution.hpp>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include "nvexec/stream_context.cuh"
#include "exec/on.hpp"
#include "timer.hpp"
#include "benchmark.hpp"

int main(int argc, char* argv[]) {
  int rank, size;
  int required = MPI_THREAD_MULTIPLE;
  int provided;
  ::MPI_Init_thread(&argc, &argv, required, &provided);
  ::MPI_Comm_size(MPI_COMM_WORLD, &size);
  ::MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  #if defined(_NVHPC_CUDA) || defined(__CUDACC__)
    int count;
    int id;

    cudaGetDeviceCount(&count);
    cudaSetDevice(rank % count);
    cudaGetDevice(&id);
    printf("Process%d running on GPU%d\n", rank, id);
  #endif

  std::vector<Timer*> timers;

  // Declare timers
  defineTimers(timers);

  // Declare schedulers
  #if defined(ENABLE_OPENMP)
    exec::static_thread_pool pool{std::thread::hardware_concurrency()};
    auto scheduler = pool.get_scheduler();
  #else
    nvexec::stream_context stream_ctx{};
    auto scheduler = stream_ctx.get_scheduler();
  #endif

  // Warmup
  comm_task(scheduler, size, timers[Dummy]);
  commh2h_task(scheduler, size, timers[Dummy]);
  transpose_task(scheduler, size, timers[Dummy]);
  axpy_task(scheduler, size, timers[Dummy]);
  sync_comm_transpose(scheduler, size, timers[Dummy]);
  async_comm_transpose(scheduler, size, timers[Dummy]);
  async_commh2h_transpose(scheduler, size, timers[Dummy]);
  sync_comm_bulk(scheduler, size, timers[Dummy]);
  async_comm_bulk(scheduler, size, timers[Dummy]);
 
  // Performance measurement
  comm_task(scheduler, size, timers[Comm]);
  commh2h_task(scheduler, size, timers[CommH2H]);
  transpose_task(scheduler, size, timers[Transpose]);
  axpy_task(scheduler, size, timers[Axpy]);
  sync_comm_transpose(scheduler, size, timers[Sync_Comm_Then]);
  async_comm_transpose(scheduler, size, timers[Async_Comm_Then]);
  async_commh2h_transpose(scheduler, size, timers[Async_CommH2H_Then]);
  sync_comm_bulk(scheduler, size, timers[Sync_Comm_Bulk]);
  async_comm_bulk(scheduler, size, timers[Async_Comm_Bulk]);

  if(rank == 0) {
    printTimers(timers);
  }
  freeTimers(timers);

  ::MPI_Finalize();
}
