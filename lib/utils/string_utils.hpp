#ifndef __STRING_UTILS_HPP__
#define __STRING_UTILS_HPP__

#include <string>
#include <sstream>
#include <iomanip>

namespace Impl {
  inline std::string zfill(int n, int length=4) {
    std::ostringstream ss;
    ss << std::setfill('0') << std::setw(length) << static_cast<int>(n);
    return ss.str();
  }

  inline std::string trimRight(const std::string &s, const std::string &c) {
    std::string str(s);
    return str.erase(s.find_last_not_of(c)+1);
  }
  
  inline std::string trimLeft(const std::string &s, const std::string &c) {
    std::string str(s);
    return str.erase(0, s.find_first_not_of(c));
  }
  
  inline std::string trim(const std::string &s, const std::string &c) {
    return trimLeft(trimRight(s, c), c);
  }
};

#endif
