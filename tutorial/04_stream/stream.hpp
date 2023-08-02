#ifndef __STREAM_HPP__
#define __STREAM_HPP__

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <experimental/mdspan>

template <typename RealType>
struct init_functor{
  #if defined(ENABLE_OPENMP)
    using Vector = thrust::host_vector<RealType>;
  #else
    using Vector = thrust::device_vector<RealType>;
  #endif
  const RealType start_A_;
  const RealType start_B_;
  const RealType start_C_;
  RealType *ptr_a_;
  RealType *ptr_b_;
  RealType *ptr_c_;

  init_functor(const RealType start_A,
               const RealType start_B,
               const RealType start_C,
               Vector& a,
               Vector& b,
               Vector& c) : start_A_(start_A), start_B_(start_B), start_C_(start_C) {
    ptr_a_ = (RealType *)thrust::raw_pointer_cast(a.data());
    ptr_b_ = (RealType *)thrust::raw_pointer_cast(b.data());
    ptr_c_ = (RealType *)thrust::raw_pointer_cast(c.data());
  }

  MDSPAN_FORCE_INLINE_FUNCTION
  void operator()(const std::size_t idx) const {
    ptr_a_[idx] = start_A_;
    ptr_b_[idx] = start_B_;
    ptr_c_[idx] = start_C_;
  }
};

template <typename RealType>
struct copy_functor{
  #if defined(ENABLE_OPENMP)
    using Vector = thrust::host_vector<RealType>;
  #else
    using Vector = thrust::device_vector<RealType>;
  #endif
  RealType *ptr_a_;
  RealType *ptr_c_;

  copy_functor(const Vector& a,
               Vector& c) {
    ptr_a_ = (RealType *)thrust::raw_pointer_cast(a.data());
    ptr_c_ = (RealType *)thrust::raw_pointer_cast(c.data());
  }

  MDSPAN_FORCE_INLINE_FUNCTION
  void operator()(const std::size_t idx) const {
    ptr_c_[idx] = ptr_a_[idx];
  }
};

template <typename RealType>
struct mul_functor{
  #if defined(ENABLE_OPENMP)
    using Vector = thrust::host_vector<RealType>;
  #else
    using Vector = thrust::device_vector<RealType>;
  #endif
  const RealType scalar_;
  RealType *ptr_b_;
  RealType *ptr_c_;

  mul_functor(const RealType scalar,
              const Vector& b,
              Vector& c) : scalar_(scalar) {
    ptr_b_ = (RealType *)thrust::raw_pointer_cast(b.data());
    ptr_c_ = (RealType *)thrust::raw_pointer_cast(c.data());
  }

  MDSPAN_FORCE_INLINE_FUNCTION
  void operator()(const std::size_t idx) const {
    ptr_b_[idx] = scalar_ * ptr_c_[idx];
  }
};

template <typename RealType>
struct add_functor{
  #if defined(ENABLE_OPENMP)
    using Vector = thrust::host_vector<RealType>;
  #else
    using Vector = thrust::device_vector<RealType>;
  #endif
  RealType *ptr_a_;
  RealType *ptr_b_;
  RealType *ptr_c_;

  add_functor(const Vector& a,
              const Vector& b,
              Vector& c) {
    ptr_a_ = (RealType *)thrust::raw_pointer_cast(a.data());
    ptr_b_ = (RealType *)thrust::raw_pointer_cast(b.data());
    ptr_c_ = (RealType *)thrust::raw_pointer_cast(c.data());
  }

  MDSPAN_FORCE_INLINE_FUNCTION
  void operator()(const std::size_t idx) const {
    ptr_c_[idx] = ptr_b_[idx] + ptr_a_[idx];
  }
};

template <typename RealType>
struct triad_functor{
  #if defined(ENABLE_OPENMP)
    using Vector = thrust::host_vector<RealType>;
  #else
    using Vector = thrust::device_vector<RealType>;
  #endif
  const RealType scalar_;
  RealType *ptr_a_;
  RealType *ptr_b_;
  RealType *ptr_c_;

  triad_functor(const RealType scalar,
                const Vector& a,
                const Vector& b,
                Vector& c) : scalar_(scalar) {
    ptr_a_ = (RealType *)thrust::raw_pointer_cast(a.data());
    ptr_b_ = (RealType *)thrust::raw_pointer_cast(b.data());
    ptr_c_ = (RealType *)thrust::raw_pointer_cast(c.data());
  }

  MDSPAN_FORCE_INLINE_FUNCTION
  void operator()(const std::size_t idx) const {
    ptr_a_[idx] = ptr_b_[idx] + scalar_ * ptr_c_[idx];
  }
};

template <typename RealType>
struct dot_functor{
  RealType *ptr_buffer_;
  RealType *ptr_a_;
  RealType *ptr_b_;

  dot_functor(thrust::device_vector<RealType>& buffer,
              const thrust::device_vector<RealType>& a,
              const thrust::device_vector<RealType>& b) {
    ptr_buffer_ = (RealType *)thrust::raw_pointer_cast(buffer.data());
    ptr_a_ = (RealType *)thrust::raw_pointer_cast(a.data());
    ptr_b_ = (RealType *)thrust::raw_pointer_cast(b.data());
  }

  MDSPAN_FORCE_INLINE_FUNCTION
  void operator()(const std::size_t idx) const {
    ptr_buffer_[idx] = ptr_a_[idx] * ptr_b_[idx];
  }
};

#endif
