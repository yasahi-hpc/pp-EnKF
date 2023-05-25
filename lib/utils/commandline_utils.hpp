#ifndef __COMMANDLINE_UTILS_HPP__
#define __COMMANDLINE_UTILS_HPP__

#include "string_utils.hpp"
#include <map>
#include <vector>
#include <string>
#include <cassert>

namespace Impl {
  using dict = std::map<std::string, std::string>;

  dict parse(int argc, char* argv[]) {
    dict kwargs;
    const std::vector<std::string> args(argv+1, argv+argc);
  
    assert(args.size() % 2 == 0);
  
    for (auto i=0; i<args.size(); i+=2) {
      std::string key = trimLeft(args[i], "-");
      std::string value = args[i+1];
      kwargs[key] = value;
    }
    return kwargs;
  }

  std::string get(dict& kwargs, const std::string& key, const std::string& default_value="") {
    if(kwargs.find(key) != kwargs.end()) {
      return kwargs[key];
    } else {
      return default_value;
    }
  }
};

#endif
