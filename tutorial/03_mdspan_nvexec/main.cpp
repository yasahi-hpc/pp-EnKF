#include <iostream>
#include <stdexec/execution.hpp>
#include <experimental/mdspan>
#include "nvexec/stream_context.cuh"

using stdexec::just; using stdexec::on;
using stdexec::sender; using stdexec::schedule;
using stdexec::then; using stdexec::sync_wait;
using stdexec::run_loop; using stdexec::get_scheduler;
using stdexec::let_value; using stdexec::when_all;
using stdexec::get_stop_token;

int main(int argc, char *argv[]) {
  // Declare a CUDA stream
  nvexec::stream_context stream_ctx{};

  auto sched = stream_ctx.get_scheduler();
  sender auto begin = schedule(sched);
  sender auto hi_again = then(begin, [] {
    return 13;
  });

  sender auto add_42 = then(hi_again, [](int arg) {return arg + 42;});
  auto [i] = sync_wait(std::move(add_42)).value();
  std::cout << "Result: " << i << std::endl;

  // Sync_wait provides a run_loop scheduler
  std::tuple<run_loop::__scheduler> t =
    sync_wait(get_scheduler()).value();
  (void) t;
  
  auto y = let_value(get_scheduler(), [](auto sched){
    return on(sched, then(just(), []{std::cout << "from run_loop\n";return 42;}));
  });
  sync_wait(std::move(y));
  
  sync_wait(when_all(just(42), get_scheduler(), get_stop_token()));
}
