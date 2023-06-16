#ifndef __IO_CONFIG_HPP__
#define __IO_CONFIG_HPP__

#include <string>

struct IOConfig {
  // File and directory names
  std::string base_dir_;
  std::string case_name_;
  std::string in_case_name_;
};

#endif
