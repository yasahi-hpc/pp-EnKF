#ifndef __EXECUTORS_VIEW_HPP__
#define __EXECUTORS_VIEW_HPP__

#include <experimental/mdspan>
#include <type_traits>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <string>
#include <numeric>
#include <cassert>
#include <algorithm>

/* [TO DO] Check the behaviour of thrust::device_vector if it is configured for CPUs */
template <typename ElementType>
  using host_vector = typename thrust::host_vector<ElementType>;
#if ! defined(ENABLE_OPENMP) && (defined(_NVHPC_CUDA) || defined(__CUDACC__) || defined(__HIPCC__))
  template <typename ElementType>
    using device_vector = typename thrust::device_vector<ElementType>;
#else
  template <typename ElementType>
    using device_vector = typename thrust::host_vector<ElementType>;
#endif

namespace stdex = std::experimental;

template <
  class ElementType,
  class Extents,
  class LayoutPolicy = stdex::layout_right
>
class View {
public:
  using mdspan_type = stdex::mdspan<ElementType, Extents, LayoutPolicy>;
  using host_vector_type = host_vector<ElementType>;
  using device_vector_type = device_vector<ElementType>;
  using value_type = typename mdspan_type::value_type;
  using extents_type = typename mdspan_type::extents_type;
  using size_type = typename mdspan_type::size_type;
  using int_type = int;
  using layout_type = typename mdspan_type::layout_type;

private:
  std::string name_;
  bool is_empty_;
  size_type size_;

  ElementType* host_data_;
  ElementType* device_data_;
  host_vector_type host_vector_;
  device_vector_type device_vector_;
  extents_type extents_;

public:
  View() : name_("empty"), is_empty_(true), host_data_(nullptr), device_data_(nullptr), size_(0) {}
  View(const std::string name, std::array<size_type, extents_type::rank()> extents)
    : name_(name), is_empty_(false), host_data_(nullptr), device_data_(nullptr), size_(0) {
    init(extents);
  }
  
  template <typename... I>
  View(const std::string name, I... indices)
    : name_(name), is_empty_(false), host_data_(nullptr), device_data_(nullptr), size_(0) {
    std::array<size_type, extents_type::rank()> extents = {static_cast<size_type>(indices)...};
    init(extents);
  }

  ~View() {}

  // Copy constructor and assignment operators
  View(const View& rhs) {
    shallow_copy(rhs);
  }
 
  View& operator=(const View& rhs) {
    if (this == &rhs) return *this;
    shallow_copy(rhs);
    return *this;
  }

  // Move and Move assignment
  View(View&& rhs) noexcept {
    deep_copy(std::forward<View>(rhs));
  }

  View& operator=(View&& rhs) {
    if (this == &rhs) return *this;
    deep_copy(std::forward<View>(rhs));
    return *this;
  }

private:
  void init(std::array<size_type, extents_type::rank()> extents) {
    auto size = std::accumulate(extents.begin(), extents.end(), 1, std::multiplies<>());
    size_ = size;
    host_vector_.resize(size_, 0);
    host_data_ = (value_type *)thrust::raw_pointer_cast(host_vector_.data());
    extents_ = extents;

    #if ! defined(ENABLE_OPENMP) && (defined(_NVHPC_CUDA) || defined(__CUDACC__) || defined(__HIPCC__))
      device_vector_.resize(size);
      device_data_ = (value_type *)thrust::raw_pointer_cast(device_vector_.data());
    #else
      // In the host configuration, device_data_ also points the host_vector
      device_data_ = (value_type *)thrust::raw_pointer_cast(host_vector_.data());
    #endif
  }

  // Only copying meta data
  void shallow_copy(const View& rhs) {
    this->setName(rhs.name());
    this->setIsEmpty(rhs.is_empty());
    host_data_ = rhs.host_data_;
    device_data_ = rhs.device_data_;
    extents_ = rhs.extents_;
    size_ = rhs.size_;
  }

  void shallow_copy(View&& rhs) {
    this->setName(rhs.name());
    this->setIsEmpty(rhs.is_empty());
    host_data_ = (value_type *)thrust::raw_pointer_cast(rhs.host_vector_.data());

    #if ! defined(ENABLE_OPENMP) && (defined(_NVHPC_CUDA) || defined(__CUDACC__) || defined(__HIPCC__))
      device_data_ = (value_type *)thrust::raw_pointer_cast(rhs.device_vector_.data());
    #else
      device_data_ = (value_type *)thrust::raw_pointer_cast(rhs.host_vector_.data());
    #endif
    extents_ = rhs.extents_;
    size_ = rhs.size_;
  }

  void deep_copy(const View& rhs) {
    this->setName(rhs.name());
    this->setIsEmpty(rhs.is_empty());
    host_vector_ = rhs.host_vector_; // not a move
    host_data_ = (value_type *)thrust::raw_pointer_cast(host_vector_.data());

    #if ! defined(ENABLE_OPENMP) && (defined(_NVHPC_CUDA) || defined(__CUDACC__) || defined(__HIPCC__))
      device_vector_ = rhs.device_vector_; // not a move
      device_data_ = (value_type *)thrust::raw_pointer_cast(device_vector_.data());
    #else
      device_data_ = (value_type *)thrust::raw_pointer_cast(host_vector_.data());
    #endif
    extents_ = rhs.extents_;
    size_ = rhs.size_;
  }

  void deep_copy(View&& rhs) {
    this->setName(rhs.name());
    this->setIsEmpty(rhs.is_empty());

    host_vector_ = std::move(rhs.host_vector_);
    host_data_   = (value_type *)thrust::raw_pointer_cast(host_vector_.data());

    #if ! defined(ENABLE_OPENMP) && (defined(_NVHPC_CUDA) || defined(__CUDACC__) || defined(__HIPCC__))
      device_vector_ = std::move(rhs.device_vector_);
      device_data_ = (value_type *)thrust::raw_pointer_cast(device_vector_.data());
    #else
      device_data_ = (value_type *)thrust::raw_pointer_cast(host_vector_.data());
    #endif
    extents_ = rhs.extents_;
    size_ = rhs.size_;
  }

public:
  void swap(View& rhs) {
    assert( extents() == rhs.extents() );
    std::string name = this->name();
    bool is_empty = this->is_empty();

    this->setName(rhs.name());
    this->setIsEmpty(rhs.is_empty());

    rhs.setName(name);
    rhs.setIsEmpty(is_empty);

    thrust::swap(this->host_vector_, rhs.host_vector_);
    #if ! defined(ENABLE_OPENMP) && (defined(_NVHPC_CUDA) || defined(__CUDACC__) || defined(__HIPCC__))
      thrust::swap(this->device_vector_, rhs.device_vector_);
    #endif
    host_data_ = (value_type *)thrust::raw_pointer_cast(host_vector_.data());
    rhs.host_data_ = (value_type *)thrust::raw_pointer_cast(rhs.host_vector_.data());

    #if ! defined(ENABLE_OPENMP) && (defined(_NVHPC_CUDA) || defined(__CUDACC__) || defined(__HIPCC__))
      device_data_ = (value_type *)thrust::raw_pointer_cast(device_vector_.data());
      rhs.device_data_ = (value_type *)thrust::raw_pointer_cast(rhs.device_vector_.data());
    #else
      device_data_ = (value_type *)thrust::raw_pointer_cast(host_vector_.data());
      rhs.device_data_ = (value_type *)thrust::raw_pointer_cast(rhs.host_vector_.data());
    #endif
  }

public:
  const std::string name() const noexcept {return name_;}
  bool is_empty() const noexcept { return is_empty_; }
  constexpr size_t rank() noexcept { return extents_type::rank(); }
  constexpr size_t rank_dynamic() noexcept { return extents_type::rank_dynamic(); }
  constexpr size_type size() const noexcept { return size_; }
  constexpr extents_type extents() const noexcept { return extents_; }
  constexpr size_type extent(size_t r) const noexcept { return extents_.extent(r); }

  value_type *data() { return device_data_; }
  const value_type *data() const { return device_data_; }
  value_type *host_data() { return host_data_; }
  const value_type *host_data() const { return host_data_; }
  value_type *device_data() { return device_data_; }
  const value_type *device_data() const { return device_data_; }

  mdspan_type mdspan() { return mdspan_type( device_data_, extents_ ) ; }
  const mdspan_type mdspan() const { return mdspan_type( device_data_, extents_ ) ; }
  mdspan_type host_mdspan() { return mdspan_type( host_data_, extents_ ) ; }
  const mdspan_type host_mdspan() const { return mdspan_type( host_data_, extents_ ) ; }
  mdspan_type device_mdspan() { return mdspan_type( device_data_, extents_ ) ; }
  const mdspan_type device_mdspan() const { return mdspan_type( device_data_, extents_ ) ; }

  inline void setName(const std::string &name) { name_ = name; }
  inline void setIsEmpty(bool is_empty) { is_empty_ = is_empty; }

  void updateDevice() {
    #if ! defined(ENABLE_OPENMP) && (defined(_NVHPC_CUDA) || defined(__CUDACC__) || defined(__HIPCC__))
      device_vector_ = host_vector_; 
    #endif
  }

  void updateSelf() {
    #if ! defined(ENABLE_OPENMP) && (defined(_NVHPC_CUDA) || defined(__CUDACC__) || defined(__HIPCC__))
      host_vector_ = device_vector_; 
    #endif
  } 

  void fill(const ElementType value = 0) {
    #if ! defined(ENABLE_OPENMP) && (defined(_NVHPC_CUDA) || defined(__CUDACC__) || defined(__HIPCC__))
      thrust::fill(device_vector_.begin(), device_vector_.end(), value);
      updateSelf();
    #else
      thrust::fill(host_vector_.begin(), host_vector_.end(), value);
    #endif
  }

  template <typename... I>
  inline ElementType& operator()(I... indices) noexcept {
    static_assert(sizeof...(I) == extents_type::rank(), "The number of indices must be equal to rank");
    return host_mdspan()(indices...);
  }

  template <typename... I>
  inline ElementType operator()(I... indices) const noexcept {
    static_assert(sizeof...(I) == extents_type::rank(), "The number of indices must be equal to rank");
    return host_mdspan()(indices...);
  }

  template <typename... I>
  inline ElementType& operator[](I... indices) noexcept {
    static_assert(sizeof...(I) == extents_type::rank(), "The number of indices must be equal to rank");
    return host_mdspan()(indices...);
  }

  template <typename... I>
  inline ElementType& operator[](I... indices) const noexcept {
    static_assert(sizeof...(I) == extents_type::rank(), "The number of indices must be equal to rank");
    return host_mdspan()(indices...);
  }
};

#endif
