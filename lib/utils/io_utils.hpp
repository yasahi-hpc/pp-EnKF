#ifndef __IO_UTILS_HPP__
#define __IO_UTILS_HPP__

#include <string>
#include <fstream>

namespace Impl {
  template <class ViewType>
  void to_binary(const std::string& filename, const ViewType& view) {
    std::FILE *fp = std::fopen(filename.c_str(), "wb");
    assert( fp != nullptr );
    using value_type = ViewType::value_type;
    std::size_t size = view.size();
    std::size_t fwrite_size = std::fwrite(view.data_handle(), sizeof(value_type), size, fp);
    assert( fwrite_size == size );
    std::fclose(fp);
  }
  
  template <class ViewType>
  void from_binary(const std::string& filename, ViewType& view) {
    auto file = std::ifstream(filename, std::ios::binary);
    assert( file.is_open() );
    using value_type = ViewType::value_type;
    std::size_t size = view.size();
    auto* data = view.data_handle();
    file.read(reinterpret_cast<char*>(data), sizeof(value_type) * size);
  }
};

#endif
