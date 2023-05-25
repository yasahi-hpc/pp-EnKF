#ifndef __SYSTEM_UTILS_HPP__
#define __SYSTEM_UTILS_HPP__

/* From https://gist.github.com/hrlou/cd440c181df5f4f2d0b61b80ca13516b
 */

#include <string>
#include <sys/stat.h>

namespace Impl {
  static int do_mkdir(const std::string& path, mode_t mode) {
    struct stat st;
    if(::stat(path.c_str(), &st) != 0) {
      if(mkdir(path.c_str(), mode) != 0 && errno != EEXIST) {
        return -1;
      }
    } else if (!S_ISDIR(st.st_mode)) {
      errno = ENOTDIR;
      return -1;
    }
    return 0;
  }
  
  int mkdirs(std::string path, mode_t mode) {
    std::string build;
    for(std::size_t pos = 0; (pos = path.find('/')) != std::string::npos;) {
      build += path.substr(0, pos + 1);
      do_mkdir(build, mode);
      path.erase(0, pos + 1);
    }
    if(!path.empty()) {
      build += path;
      do_mkdir(build, mode);
    }
    return 0;
  }
  
};

#endif
