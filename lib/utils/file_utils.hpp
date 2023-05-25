#ifndef __FILE_UTILS_HPP__
#define __FILE_UTILS_HPP__

#include <string_view>
#include <fstream>

namespace Impl {
  inline bool isFileExists(const std::string file_name) {
    std::ifstream file(file_name);
    return file.good();
  }
};

#endif
