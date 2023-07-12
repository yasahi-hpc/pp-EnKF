#ifndef __IO_UTILS_HPP__
#define __IO_UTILS_HPP__

#include <string>
#include <cstring>
#include <fstream>
#include <cassert>

namespace Impl {
  // Binary files
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

  // CSV files
  template <typename IndexType>
  void to_csv(const std::string& filename,
              std::map<IndexType, std::vector<std::string>>& dict,
              const std::string& separator=",",
              const bool header=true,
              const bool index=false) {
    std::ofstream file(filename);
    assert( file.is_open() );
    for(auto d: dict) {
      if(!header) continue;
      auto key = d.first;
      auto value = d.second;
      if(index) file << key << separator;

      for(std::size_t i=0; i<value.size(); i++) {
        if(i != value.size()-1) {
          file << value[i] << separator;
        } else {
          file << value[i] << std::endl;
        }
      }

      /*
      for(auto v: value) {
        if(v != value.back()) {
          file << v << separator;
        } else {
          file << v << std::endl;
        }
      }
      */
    }
  }

  template <typename IndexType>
  void from_csv(const std::string& filename,
                std::map<IndexType, std::vector<std::string>>& dict,
                const std::string& separator=",",
                const bool header=true,
                const bool index=false) {
    std::ifstream file(filename, std::ios::in);
    assert( file.is_open() );
    std::string line, word;
    std::vector<std::string> row;
    int count = 0;
    while(std::getline(file, line)) {
      row.clear();
      std::stringstream str(line);
      char* delim = new char[separator.size()];
      std::strcpy(delim, separator.c_str());
      while(std::getline(str, word, *delim)) {
        row.push_back(word);
      }
      delete[] delim;
 
      if(index) {
        int key = stoi(row.at(0));
        row.erase(row.begin());
        dict[key] = row;
      } else {
        int key = count;
        dict[key] = row;
      }
      count += 1;
    }
  }
};

#endif
