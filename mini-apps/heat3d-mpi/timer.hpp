#ifndef __TIMER_HPP__
#define __TIMER_HPP__

#include <chrono>
#include <vector>
#include <iostream>
#include <map>

struct Timer {
private:
  std::string label_;
  double accumulated_time_;
  int calls_;
  bool use_time_stamps_;
  const int max_counts_ = 10000;
  std::chrono::high_resolution_clock::time_point init_, begin_, end_;
  std::vector<std::chrono::high_resolution_clock::time_point> begin_points_;
  std::vector<std::chrono::high_resolution_clock::time_point> end_points_;

public:
  Timer() : use_time_stamps_(false), accumulated_time_(0.0), calls_(0), label_("") {
    init_ = std::chrono::high_resolution_clock::now();
  };

  Timer(const std::string label) : use_time_stamps_(false), accumulated_time_(0.0), calls_(0), label_(label) {
    init_ = std::chrono::high_resolution_clock::now();
  };

  Timer(const std::string label, bool use_time_stamps) : use_time_stamps_(use_time_stamps), accumulated_time_(0.0), calls_(0), label_(label) {
    init_ = std::chrono::high_resolution_clock::now();
    begin_points_.reserve(max_counts_);
    end_points_.reserve(max_counts_);
  };

  virtual ~Timer(){};

  void begin() {
    begin_ = std::chrono::high_resolution_clock::now();
    if(use_time_stamps_) begin_points_.push_back(begin_);
  }

  void end() {
    end_ = std::chrono::high_resolution_clock::now();
    if(use_time_stamps_) end_points_.push_back(end_);
    accumulated_time_ += std::chrono::duration_cast<std::chrono::duration<double> >(end_ - begin_).count();
    calls_++;
  }

  auto getTimeStamps(const std::vector<std::chrono::high_resolution_clock::time_point>& points) {
    std::vector<double> time_stamps;
    time_stamps.reserve(points.size());
    for(const auto &point : points) {
      double elapsed_time = std::chrono::duration_cast<std::chrono::duration<double> >(point - init_).count();
      time_stamps.push_back(elapsed_time);
    }
    return time_stamps;
  }

  auto beginPoints() { return getTimeStamps(begin_points_); }
  auto endPoints() { return getTimeStamps(end_points_); }

  double seconds(){return accumulated_time_;};
  double milliseconds(){return accumulated_time_*1.e3;};
  int calls(){return calls_;};
  std::string label(){return label_;};
  bool use_time_stamps(){return use_time_stamps_;};
  void reset(){accumulated_time_ = 0.; calls_ = 0;};
  void reset(const std::chrono::high_resolution_clock::time_point init){
    init_ = init;
    accumulated_time_ = 0.; calls_ = 0;
  };
};

enum TimerEnum : int {Total,
                      MainLoop,
                      Heat,
                      HaloPack,
                      HaloUnpack,
                      HaloComm,
                      IO,
                      Nb_timers};

static void defineTimers(std::vector<Timer*> &timers, bool use_time_stamps=false) {
  // Set timers
  timers.resize(Nb_timers);
  timers[TimerEnum::Total]      = new Timer("total");
  timers[TimerEnum::MainLoop]   = new Timer("MainLoop");
  timers[TimerEnum::Heat]       = new Timer("Heat", use_time_stamps);
  timers[TimerEnum::HaloPack]   = new Timer("HaloPack", use_time_stamps);
  timers[TimerEnum::HaloUnpack] = new Timer("HaloUnpack", use_time_stamps);
  timers[TimerEnum::HaloComm]   = new Timer("HaloComm", use_time_stamps);
  timers[TimerEnum::IO]         = new Timer("IO");
}

static void printTimers(std::vector<Timer*> &timers) {
  // Print timer information
  for(auto it = timers.begin(); it != timers.end(); ++it) {
    std::cout << (*it)->label() << " " << (*it)->seconds() << " [s], " << (*it)->calls() << " calls" << std::endl;
  }
}

static void resetTimers(std::vector<Timer*> &timers) {
  auto init = std::chrono::high_resolution_clock::now();
  for(auto it = timers.begin(); it != timers.end(); ++it) {
    (*it)->reset(init);
  }
};

static void freeTimers(std::vector<Timer*> &timers) {
  for(auto it = timers.begin(); it != timers.end(); ++it) {
    delete *it;
  }
};

inline auto timersToDict(std::vector<Timer*> &timers) {
  std::map<int, std::vector<std::string> > dict;

  // Header
  int key = 0;
  dict[key] = std::vector<std::string>{"name", "seconds", "count"};
  for(auto it = timers.begin(); it != timers.end(); ++it) {
    key++;
    std::vector<std::string> value;
    value.push_back( (*it)->label() );
    value.push_back( std::to_string( (*it)->seconds() ) );
    value.push_back( std::to_string( (*it)->calls() ) );
    dict[key] = value;
  }
  return dict;
};

inline auto timeStampsToDict(std::vector<Timer*> &timers) {
  std::map<std::string, std::vector<double> > stamp_dict;
  for(auto it = timers.begin(); it != timers.end(); ++it) {
    std::string label = (*it)->label();
    if((*it)->use_time_stamps()) {
      auto begins = (*it)->beginPoints();
      auto ends   = (*it)->endPoints();

      std::string begin_label = label + "_begin";
      std::string end_label   = label + "_end";
      stamp_dict[begin_label] = begins;
      stamp_dict[end_label]   = ends;
    }
  }

  std::vector<std::string> header;
  std::map<int, std::vector<std::string> > dict;

  // Initialize dict
  auto stamp_size = stamp_dict.size();
  for(auto item : stamp_dict) {
    auto key   = item.first;
    auto value = item.second;
    header.push_back(key);

    for(std::size_t i=0; i<value.size(); i++) {
      std::vector<std::string> empty(stamp_size);
      dict[i+1] = empty;
    }
  }

  // Copy header and construct dict
  dict[0] = header;
  for(std::size_t idx=0; idx<header.size(); idx++) {
    auto key = header[idx];
    auto value = stamp_dict[key];
    for(std::size_t i=0; i<value.size(); i++) {
      dict[i+1].at(idx) = std::to_string(value[i]);
    }
  }

  return dict;
};


template < class FunctorType >
void exec_with_timer(FunctorType&& f, Timer *timer) {
  timer->begin();
  f();
  timer->end();
}

#endif
