#ifndef __TIMER_HPP__
#define __TIMER_HPP__

#include <chrono>
#include <vector>
#include <iostream>

struct Timer {
private:
  std::string label_;
  double accumulated_time_;
  int calls_;
  std::chrono::high_resolution_clock::time_point begin_, end_;

public:
  Timer() : accumulated_time_(0.0), calls_(0), label_(""){};
  Timer(const std::string label) : accumulated_time_(0.0), calls_(0), label_(label){};
  virtual ~Timer(){};

  void begin() {
    begin_ = std::chrono::high_resolution_clock::now();
  }

  void end() {
    end_ = std::chrono::high_resolution_clock::now();
    accumulated_time_ += std::chrono::duration_cast<std::chrono::duration<double> >(end_ - begin_).count();
    calls_++;
  }

  double seconds(){return accumulated_time_;};
  double milliseconds(){return accumulated_time_*1.e3;};
  int calls(){return calls_;};
  std::string label(){return label_;};
  void reset(){accumulated_time_ = 0.; calls_ = 0;};
};

enum TimerEnum : int {Comm,
                      CommH2H,
                      Transpose,
                      Axpy,
                      Sync_Comm_Then,
                      Async_Comm_Then,
                      Async_CommH2H_Then,
                      Sync_Comm_Bulk,
                      Async_Comm_Bulk,
                      Dummy,
                      Nb_timers};

static void defineTimers(std::vector<Timer*> &timers) {
  // Set timers
  timers.resize(Nb_timers);
  timers[TimerEnum::Comm]               = new Timer("comm");
  timers[TimerEnum::CommH2H]            = new Timer("commh2h");
  timers[TimerEnum::Transpose]          = new Timer("transpose");
  timers[TimerEnum::Axpy]               = new Timer("axpy");
  timers[TimerEnum::Sync_Comm_Then]     = new Timer("sync_comm_then");
  timers[TimerEnum::Async_Comm_Then]    = new Timer("Async_comm_then");
  timers[TimerEnum::Async_CommH2H_Then] = new Timer("Async_commh2h_then");
  timers[TimerEnum::Sync_Comm_Bulk]     = new Timer("Sync_comm_bulk");
  timers[TimerEnum::Async_Comm_Bulk]    = new Timer("Async_comm_bulk");
  timers[TimerEnum::Dummy]              = new Timer("Dummy");
}

static void printTimers(std::vector<Timer*> &timers) {
  // Print timer information
  for(auto it = timers.begin(); it != timers.end(); ++it) {
    std::cout << (*it)->label() << " " << (*it)->seconds() << " [s], " << (*it)->calls() << " calls" << std::endl;
  }
}

static void resetTimers(std::vector<Timer*> &timers) {
  for(auto it = timers.begin(); it != timers.end(); ++it) {
    (*it)->reset();
  }
};

static void freeTimers(std::vector<Timer*> &timers) {
  for(auto it = timers.begin(); it != timers.end(); ++it) {
    delete *it;
  }
};

template < class FunctorType >
void exec_with_timer(FunctorType&& f, Timer *timer) {
  timer->begin();
  f();
  timer->end();
}

#endif
