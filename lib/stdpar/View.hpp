#ifndef __STDPAR_VIEW_HPP__
#define __STDPAR_VIEW_HPP__

#include <experimental/mdspan>
#include <type_traits>
#include <vector>
#include <string>
#include <numeric>
#include <cassert>
#include <execution>
#include <algorithm>

namespace stdex = std::experimental;

template <
  class ElementType,
  class Extents,
  class LayoutPolicy = std::experimental::layout_right
>
class View {
public:
  using mdspan_type = stdex::mdspan<ElementType, Extents, LayoutPolicy>;
  using vector_type = std::vector<ElementType>;
  using value_type = typename mdspan_type::value_type;
  using extents_type = typename mdspan_type::extents_type;
  using size_type = typename mdspan_type::size_type;
  using int_type = int;
  using layout_type = typename mdspan_type::layout_type;

private:
  std::string name_;
  bool is_empty_;
  size_type size_;

  ElementType* data_;
  vector_type vector_;
  extents_type extents_;

public:
  View() : name_("empty"), is_empty_(true), data_(nullptr), size_(0) {}
  View(const std::string name, std::array<size_type, extents_type::rank()> extents)
    : name_(name), is_empty_(false), data_(nullptr), size_(0) {
    init(extents);
  }
  
  template <typename... I, 
             std::enable_if_t<
               std::is_integral_v< 
                 std::tuple_element_t<0, std::tuple<I...>>
               >, std::nullptr_t> = nullptr>
  View(const std::string name, I... indices)
    : name_(name), is_empty_(false), data_(nullptr), size_(0) {
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
    vector_.resize(size_, 0);
    data_ = vector_.data();
    extents_ = extents;
  }

  // Only copying meta data
  void shallow_copy(const View& rhs) {
    this->setName(rhs.name()+"_copy");
    this->setIsEmpty(rhs.is_empty());
    data_ = rhs.data_;
    extents_ = rhs.extents_;
    size_ = rhs.size_;
  }

  void shallow_copy(View&& rhs) {
    this->setName(rhs.name()+"_copy");
    this->setIsEmpty(rhs.is_empty());
    data_ = rhs.vector_.data();
    extents_ = rhs.extents_;
    size_ = rhs.size_;
  }

  void deep_copy(const View& rhs) {
    this->setName(rhs.name());
    this->setIsEmpty(rhs.is_empty());
    vector_ = rhs.vector_; // not a move
    data_ = vector_.data();
    extents_ = rhs.extents_;
    size_ = rhs.size_;
  }

  void deep_copy(View&& rhs) {
    this->setName(rhs.name());
    this->setIsEmpty(rhs.is_empty());
    vector_ = std::move(rhs.vector_);
    data_ = vector_.data();
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

    std::swap(this->vector_, rhs.vector_);
    data_ = vector_.data();
    rhs.data_ = rhs.vector_.data();
  }

public:
  const std::string name() const noexcept {return name_;}
  bool is_empty() const noexcept { return is_empty_; }
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

  // Do nothing, in order to inform compiler "vector_" is on device by launching a gpu kernel
  // [TO DO] this method may cause a problem if called from shallow copied view
  void updateDevice() {
    auto tmp = vector_;
    std::copy(std::execution::par_unseq, tmp.begin(), tmp.end(), vector_.begin());
  }

  // Do nothing, in order to inform compiler "vector_" is on host by launching a host kernel
  // [TO DO] this method may cause a problem if called from shallow copied view
  void updateSelf() {
    auto tmp = vector_;
    std::copy(tmp.begin(), tmp.end(), vector_.begin());
  } 

  void fill(const ElementType value = 0) {
    std::fill(vector_.begin(), vector_.end(), value);
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
