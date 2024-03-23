#ifndef SYCL_VIEW_HPP
#define SYCL_VIEW_HPP

#include <type_traits>
#include <vector>
#include <string>
#include <numeric>
#include <cassert>
#include <execution>
#include <algorithm>
#include <memory>
#include <sycl/sycl.hpp>
#include <experimental/mdspan>

namespace stdex = std::experimental;

template <
  class ElementType,
  class Extents,
  class LayoutPolicy = std::experimental::layout_right
>
class View {
public:
  using mdspan_type = stdex::mdspan<ElementType, Extents, LayoutPolicy>;
  using value_type = typename mdspan_type::value_type;
  using extents_type = typename mdspan_type::extents_type;
  using size_type = typename mdspan_type::size_type;
  using int_type = int;
  using layout_type = typename mdspan_type::layout_type;

private:
  std::string name_;
  bool is_empty_;
  bool is_copied_;
  size_type size_;

  value_type* data_;
  extents_type extents_;

  // SYCL objects
  sycl::queue q_;

public:
  View() : q_(sycl::default_selector_v),  name_("empty"), is_empty_(true), is_copied_(false), data_(nullptr), size_(0) {}

  View(sycl::queue& q, const std::string name, std::array<size_type, extents_type::rank()> extents)
    : q_(q), name_(name), is_empty_(false), is_copied_(false), data_(nullptr), size_(0) {
    init(extents);
  }

  template <typename... I,
             std::enable_if_t<
               std::is_integral_v<
                 std::tuple_element_t<0, std::tuple<I...>>
               >, std::nullptr_t> = nullptr>
  View(sycl::queue& q, const std::string name, I... indices)
    : q_(q), name_(name), is_empty_(false), is_copied_(false), data_(nullptr), size_(0) {
    std::array<size_type, extents_type::rank()> extents = {static_cast<size_type>(indices)...};
    init(extents);
  }

  ~View() {
    if(! is_copied_ && ! is_empty_) {
      if(data_ != nullptr) { sycl::free(data_, q_); }
    }
  }

  // Copy constructor and assignment operators
  View(const View& rhs) : q_(rhs.q_), name_(rhs.name_), data_(rhs.data_), extents_(rhs.extents_), size_(rhs.size_) {
    this->setIsEmpty(rhs.is_empty());
    this->setIsCopied(true);
  }

  View& operator=(const View& rhs) {
    if (this == &rhs) return *this;
    this->setName(rhs.name());
    this->setIsEmpty(rhs.is_empty());
    this->setIsCopied(true);

    q_    = rhs.q_;
    data_ = rhs.data_;
    extents_ = rhs.extents_;
    size_ = rhs.size_;
    return *this;
  }

  // Move and Move assignment
  View(View&& rhs) noexcept : q_(rhs.q_), name_(rhs.name_), data_(std::move(rhs.data_)), extents_(rhs.extents_), size_(rhs.size_) {
    this->setIsEmpty(rhs.is_empty());
    this->setIsCopied(false);

    rhs.name_ = "";
    rhs.size_ = 0;
    rhs.setIsCopied(true);
    rhs.data_ = nullptr;
  }

  View& operator=(View&& rhs) {
    if (this == &rhs) return *this;
    this->setName(rhs.name());
    this->setIsEmpty(rhs.is_empty());
    this->setIsCopied(false);
    q_ = rhs.q_;
    data_ = std::move(rhs.data_);
    extents_ = std::move(rhs.extents_);
    size_ = std::move(rhs.size_);

    rhs.name_ = "";
    rhs.size_ = 0;
    rhs.setIsCopied(true);
    rhs.data_ = nullptr;
    return *this;
  }

private:
  void init(std::array<size_type, extents_type::rank()> extents) {
    auto size = std::accumulate(extents.begin(), extents.end(), 1, std::multiplies<>());
    size_ = size;
    data_ = sycl::malloc_shared<value_type>(size_, q_);
    extents_ = extents;
  }

public:
  void swap(View& rhs) {
    assert( extents() == rhs.extents() );
    std::string name = this->name();
    bool is_empty = this->is_empty();
    bool is_copied = this->is_copied();

    this->setName(rhs.name());
    this->setIsEmpty(rhs.is_empty());
    this->setIsCopied(rhs.is_copied());

    rhs.setName(name);
    rhs.setIsEmpty(is_empty);
    rhs.setIsCopied(is_copied);

    std::swap(data_, rhs.data_);
    std::swap(q_, rhs.q_);
  }

public:
  const std::string name() const noexcept {return name_;}
  bool is_empty() const noexcept { return is_empty_; }
  bool is_copied() const noexcept { return is_copied_; }
  constexpr size_t rank() noexcept { return extents_type::rank(); }
  constexpr size_t rank_dynamic() noexcept { return extents_type::rank_dynamic(); }
  constexpr size_type size() const noexcept { return size_; }
  constexpr extents_type extents() const noexcept { return extents_; }
  constexpr size_type extent(size_t r) const noexcept { return extents_.extent(r); }

  value_type *data() { return data_; }
  const value_type *data() const { return data_; }
  mdspan_type mdspan() { return mdspan_type( data_, extents_ ) ; }
  const mdspan_type mdspan() const { return mdspan_type( data_, extents_ ) ; }

  inline void setName(const std::string &name) { name_ = name; }
  inline void setIsEmpty(bool is_empty) { is_empty_ = is_empty; }
  inline void setIsCopied(bool is_copied) { is_copied_ = is_copied; }

  // Do nothing, in order to inform compiler "vector_" is on device by launching a gpu kernel
  // [TO DO] this method may cause a problem if called from shallow copied view
  void updateDevice() {
    //auto tmp = vector_;
    //std::copy(std::execution::par_unseq, tmp.begin(), tmp.end(), vector_.begin());
  }

  // Do nothing, in order to inform compiler "vector_" is on host by launching a host kernel
  // [TO DO] this method may cause a problem if called from shallow copied view
  void updateSelf() {
    //auto tmp = vector_;
    //std::copy(tmp.begin(), tmp.end(), vector_.begin());
  }

  void fill(const ElementType value = 0) {
    std::fill(data_, data_+size_, value);
  }

  template <typename... I>
  inline ElementType& operator()(I... indices) noexcept {
    static_assert(sizeof...(I) == extents_type::rank(), "The number of indices must be equal to rank");
    return mdspan()(indices...);
  }

  template <typename... I>
  inline ElementType operator()(I... indices) const noexcept {
    static_assert(sizeof...(I) == extents_type::rank(), "The number of indices must be equal to rank");
    return mdspan()(indices...);
  }

  template <typename... I>
  inline ElementType& operator[](I... indices) noexcept {
    static_assert(sizeof...(I) == extents_type::rank(), "The number of indices must be equal to rank");
    return mdspan()(indices...);
  }

  template <typename... I>
  inline ElementType& operator[](I... indices) const noexcept {
    static_assert(sizeof...(I) == extents_type::rank(), "The number of indices must be equal to rank");
    return mdspan()(indices...);
  }
};

#endif