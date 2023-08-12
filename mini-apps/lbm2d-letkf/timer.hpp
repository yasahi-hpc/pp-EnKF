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
  void reset(){accumulated_time_ = 0.; calls_ = 0;};
  void reset(const std::chrono::high_resolution_clock::time_point init){
    init_ = init;
    accumulated_time_ = 0.; calls_ = 0;
  };
};

enum TimerEnum : int {Total,
                      MainLoop,
                      DA,
                      DA_Load,
                      DA_Load_H2D,
                      DA_Load_rho,
                      DA_Load_u,
                      DA_Load_v,
                      DA_Load_H2D_rho,
                      DA_Load_H2D_u,
                      DA_Load_H2D_v,
                      DA_Pack_X,
                      DA_All2All_X,
                      DA_Unpack_X,
                      DA_Pack_Y,
                      DA_All2All_Y,
                      DA_Unpack_Y,
                      DA_Pack_Obs,
                      DA_Broadcast,
                      DA_Broadcast_rho,
                      DA_Broadcast_u,
                      DA_Broadcast_v,
                      DA_LETKF,
                      DA_Update,
                      Diag,
                      LBMSolver,
                      Nb_timers};

static void defineTimers(std::vector<Timer*> &timers, bool use_time_stamps=false) {
  // Set timers
  timers.resize(Nb_timers);
  timers[TimerEnum::Total]         = new Timer("total");
  timers[TimerEnum::MainLoop]      = new Timer("MainLoop");
  timers[TimerEnum::DA]            = new Timer("DA", use_time_stamps);
  timers[TimerEnum::DA_Load]       = new Timer("DA_Load", use_time_stamps);
  timers[TimerEnum::DA_Load_H2D]   = new Timer("DA_Load_H2D", use_time_stamps);
  timers[TimerEnum::DA_Load_rho]     = new Timer("DA_Load_rho", use_time_stamps);
  timers[TimerEnum::DA_Load_H2D_rho] = new Timer("DA_Load_H2D_rho", use_time_stamps);
  timers[TimerEnum::DA_Load_u]     = new Timer("DA_Load_u", use_time_stamps);
  timers[TimerEnum::DA_Load_H2D_u] = new Timer("DA_Load_H2D_u", use_time_stamps);
  timers[TimerEnum::DA_Load_v]     = new Timer("DA_Load_v", use_time_stamps);
  timers[TimerEnum::DA_Load_H2D_v] = new Timer("DA_Load_H2D_v", use_time_stamps);
  timers[TimerEnum::DA_Pack_X]     = new Timer("DA_Pack_X", use_time_stamps);
  timers[TimerEnum::DA_All2All_X]  = new Timer("DA_All2All_X", use_time_stamps);
  timers[TimerEnum::DA_Unpack_X]   = new Timer("DA_Unpack_X", use_time_stamps);
  timers[TimerEnum::DA_Pack_Y]     = new Timer("DA_Pack_Y", use_time_stamps);
  timers[TimerEnum::DA_All2All_Y]  = new Timer("DA_All2All_Y", use_time_stamps);
  timers[TimerEnum::DA_Unpack_Y]   = new Timer("DA_Unpack_Y", use_time_stamps);
  timers[TimerEnum::DA_Pack_Obs]   = new Timer("DA_Pack_Obs", use_time_stamps);
  timers[TimerEnum::DA_Broadcast]  = new Timer("DA_Broadcast", use_time_stamps);
  timers[TimerEnum::DA_Broadcast_rho] = new Timer("DA_Broadcast_rho", use_time_stamps);
  timers[TimerEnum::DA_Broadcast_u]  = new Timer("DA_Broadcast_u", use_time_stamps);
  timers[TimerEnum::DA_Broadcast_v]  = new Timer("DA_Broadcast_v", use_time_stamps);
  timers[TimerEnum::DA_LETKF]      = new Timer("DA_LETKF", use_time_stamps);
  timers[TimerEnum::DA_Update]     = new Timer("DA_Update", use_time_stamps);
  timers[TimerEnum::Diag]          = new Timer("diag");
  timers[TimerEnum::LBMSolver]     = new Timer("lbm");
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
    if(label.find("DA") != std::string::npos) {
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
