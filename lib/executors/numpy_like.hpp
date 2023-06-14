#ifndef __EXECUTORS_NUMPY_LIKE_HPP__
#define __EXECUTORS_NUMPY_LIKE_HPP__

#include <cmath>
#include <cassert>
#include <array>
#include <vector>
#include <experimental/mdspan>
#include "../Iteration.hpp"
#include "Parallel_For.hpp"

namespace Impl {
  template <class InputView, class OutputView,
            std::enable_if_t<InputView::rank()==3 && OutputView::rank()==2, std::nullptr_t> = nullptr>
  void mean_(InputView& in, OutputView& out, int axis) {
    // 3D -> 2D
    using value_type = InputView::value_type;
    int reduce_dim = axis==-1 ? InputView::rank()-1 : axis;
    assert(reduce_dim < InputView::rank());
    const std::size_t reduce_size = in.extent(reduce_dim);
    const std::size_t n = in.size() / reduce_size;
    const auto n0 = (reduce_dim==0) ? in.extent(1) : in.extent(0);
    const auto n1 = n / n0;

    // Not quite sure, this is a better strategy
    IteratePolicy<typename InputView::layout_type, 2> policy2d({0, 0}, {n0, n1});

    Impl::for_each(policy2d,
      [=](const int i0, const int i1) {
        value_type sum = 0;

        for(int ir=0; ir < reduce_size; ir++) {
          if(reduce_dim == 0) {
            auto sub_in = submdspan(in, ir, std::experimental::full_extent, std::experimental::full_extent);
            sum += sub_in(i0, i1);
          } else if(reduce_dim == 1) {
            auto sub_in = submdspan(in, std::experimental::full_extent, ir, std::experimental::full_extent);
            sum += sub_in(i0, i1);
          } else {
            auto sub_in = submdspan(in, std::experimental::full_extent, std::experimental::full_extent, ir);
            sum += sub_in(i0, i1);
          }
        }
        out(i0, i1) = sum / static_cast<value_type>(reduce_size);
    });
  }

  template <class InputView, class OutputView,
            std::enable_if_t<InputView::rank()==3 && OutputView::rank()==3, std::nullptr_t> = nullptr>
  void mean_(InputView& in, OutputView& out, int axis) {
    // 3D -> 3D (keepdims = true)
    int reduce_dim = axis==-1 ? InputView::rank()-1 : axis;
    assert(reduce_dim < InputView::rank());
    assert(out.extent(reduce_dim) == 1);

    if(reduce_dim == 0) {
      auto sub_out = submdspan(out, 0, std::experimental::full_extent, std::experimental::full_extent);
      mean_(in, sub_out, axis);
    } else if(reduce_dim == 1) {
      auto sub_out = submdspan(out, std::experimental::full_extent, 0, std::experimental::full_extent);
      mean_(in, sub_out, axis);
    } else {
      auto sub_out = submdspan(out, std::experimental::full_extent, std::experimental::full_extent, 0);
      mean_(in, sub_out, axis);
    }
  }

  template <class InputView, class OutputView,
            std::enable_if_t<InputView::rank()==2 && OutputView::rank()==1, std::nullptr_t> = nullptr>
  void mean_(InputView& in, OutputView& out, int axis) {
    // 2D -> 1D
    using value_type = InputView::value_type;
    int reduce_dim = axis==-1 ? InputView::rank()-1 : axis;
    assert(reduce_dim < InputView::rank());
    const std::size_t reduce_size = in.extent(reduce_dim);
    const auto n = (reduce_dim==0) ? in.extent(1) : in.extent(0);
    IteratePolicy<typename InputView::layout_type, 1> policy1d(n);

    Impl::for_each(policy1d,
      [=](const int idx) {
        value_type sum = 0;

        for(int ir=0; ir < reduce_size; ir++) {
          if(reduce_dim == 0) {
            auto sub_in = submdspan(in, ir, std::experimental::full_extent);
            sum += sub_in(idx);
          } else {
            auto sub_in = submdspan(in, std::experimental::full_extent, ir);
            sum += sub_in(idx);
          }
        }
        out(idx) = sum / static_cast<value_type>(reduce_size);
    });
  }

  template <class InputView, class OutputView,
            std::enable_if_t<InputView::rank()==2 && OutputView::rank()==2, std::nullptr_t> = nullptr>
  void mean_(InputView& in, OutputView& out, int axis) {
    // 2D -> 2D
    int reduce_dim = axis==-1 ? InputView::rank()-1 : axis;
    assert(reduce_dim < InputView::rank());
    assert(out.extent(reduce_dim) == 1);

    if(reduce_dim == 0) {
      auto sub_out = submdspan(out, 0, std::experimental::full_extent);
      mean_(in, sub_out, axis);
    } else {
      auto sub_out = submdspan(out, std::experimental::full_extent, 0);
      mean_(in, sub_out, axis);
    }
  }

  template <class InputView, class OutputView,
            std::enable_if_t<InputView::rank()==1, std::nullptr_t> = nullptr>
  void mean_(InputView& in, OutputView& out, int axis, bool keepdims) {
    // 1D -> 1D
    int reduce_dim = axis==-1 ? InputView::rank()-1 : axis;
    assert(reduce_dim < InputView::rank()); // This must be zero
    // Just parallel reduction and store it into a 0D View
  }

  // Global interface
  template <class InputView, class OutputView>
  void mean(InputView& in, OutputView& out, int axis) {
    static_assert( std::is_same_v<typename InputView::value_type, typename OutputView::value_type> );
    static_assert( std::is_same_v<typename InputView::layout_type, typename OutputView::layout_type> );
    static_assert( InputView::rank() >= OutputView::rank() );
    static_assert( InputView::rank() <= 3 );

    mean_(in, out, axis);
  }

  /* axpy */
  template <class InoutView,
            std::enable_if_t<InoutView::rank()==1, std::nullptr_t> = nullptr>
  void axpy(InoutView& x,
            const typename InoutView::value_type& y,
            const typename InoutView::value_type beta=1,
            const typename InoutView::value_type alpha=1) {

    // Inplace: 1D + 0D -> 1D (broadcasting)
    const auto n0 = x.extent(0);
    IteratePolicy<typename InoutView::layout_type, 1> policy1d(0, n0);

    Impl::for_each(policy1d,
      [=](const int i0) {
        x(i0) = alpha * x(i0) + beta * y;
      });
  }

  template <class InoutView,
            std::enable_if_t<InoutView::rank()==1, std::nullptr_t> = nullptr>
  void axpy(const InoutView& x,
            const typename InoutView::value_type& y,
            InoutView& z,
            const typename InoutView::value_type beta=1,
            const typename InoutView::value_type alpha=1) {
    // Outplace: 1D + 0D -> 1D (broadcasting)
    assert(x.extents() == z.extents());

    const auto n0 = x.extent(0);
    IteratePolicy<typename InoutView::layout_type, 1> policy1d(0, n0);

    Impl::for_each(policy1d,
      [=](const int i0) {
        z(i0) = alpha * x(i0) + beta * y;
      });
  }

  template <class InoutView,
            class InputView,
            std::enable_if_t<InoutView::rank()==1 && InputView::rank()==1, std::nullptr_t> = nullptr>
  void axpy_(InoutView& x,
             const InputView& y,
             const typename InoutView::value_type beta=1,
             const typename InoutView::value_type alpha=1) {
    // Inplace: 1D + 1D -> 1D
    const auto nx0 = x.extent(0);
    const auto ny0 = y.extent(0);

    if( x.extents() == y.extents() ) {
      IteratePolicy<typename InoutView::layout_type, 1> policy1d(0, nx0);

      Impl::for_each(policy1d,
        [=](const int i0) {
          x(i0) = alpha * x(i0) + beta * y(i0);
        });
    } else if( ny0 == 1 && ny0 < nx0 ) {
      IteratePolicy<typename InoutView::layout_type, 1> policy1d(0, nx0);
      Impl::for_each(policy1d,
        [=](const int i0) {
          x(i0) = alpha * x(i0) + beta * y(0);
        });
    } else {
      std::runtime_error("Cannot broadcast y to x. Check the shapes of x and y.");
    }
  }

  template <class InoutView,
            class InputView,
            std::enable_if_t<InoutView::rank()==1 && InputView::rank()==1, std::nullptr_t> = nullptr>
  void axpy_(const InoutView& x,
             const InputView& y,
             InoutView& z,
             const typename InoutView::value_type beta=1,
             const typename InoutView::value_type alpha=1) {
    // Outplace: 1D + 1D -> 1D
    assert(x.extents() == z.extents());
    const auto nx0 = x.extent(0);
    const auto ny0 = y.extent(0);

    if( x.extents() == y.extents() ) {
      IteratePolicy<typename InoutView::layout_type, 1> policy1d(0, nx0);

      Impl::for_each(policy1d,
        [=](const int i0) {
          z(i0) = alpha * x(i0) + beta * y(i0);
        });
    } else if( ny0 == 1 && ny0 < nx0 ) {
      IteratePolicy<typename InoutView::layout_type, 1> policy1d(0, nx0);
      Impl::for_each(policy1d,
        [=](const int i0) {
          z(i0) = alpha * x(i0) + beta * y(0);
        });
    } else {
      std::runtime_error("Cannot broadcast y to x. Check the shapes of x and y.");
    }
  }

  // Axpy 2D (maybe better to use squeeze or reshape and reuse 1D version)
  template <class InoutView,
            std::enable_if_t<InoutView::rank()==2, std::nullptr_t> = nullptr>
  void axpy(InoutView& x, 
            const typename InoutView::value_type& y,
            const typename InoutView::value_type beta=1,
            const typename InoutView::value_type alpha=1) {

    // Inplace: 2D + 0D -> 2D (broadcasting)
    const auto n0 = x.extent(0), n1 = x.extent(1);
    IteratePolicy<typename InoutView::layout_type, 2> policy2d({0, 0}, {n0, n1});

    Impl::for_each(policy2d,
      [=](const int i0, const int i1) {
        x(i0, i1) = alpha * x(i0, i1) + beta * y;
      });
  }

  template <class InoutView,
            std::enable_if_t<InoutView::rank()==2, std::nullptr_t> = nullptr>
  void axpy(const InoutView& x,
            const typename InoutView::value_type& y,
            InoutView& z,
            const typename InoutView::value_type beta=1,
            const typename InoutView::value_type alpha=1) {
    // Outplace: 2D + 0D -> 2D (broadcasting)
    assert(x.extents() == z.extents());

    const auto n0 = x.extent(0), n1 = x.extent(1);
    IteratePolicy<typename InoutView::layout_type, 2> policy2d({0, 0}, {n0, n1});
    Impl::for_each(policy2d,
      [=](const int i0, const int i1) {
        z(i0, i1) = alpha * x(i0, i1) + beta * y;
      });
  }

  template <class InoutView,
            class InputView,
            std::enable_if_t<InoutView::rank()==2 && InputView::rank()==1, std::nullptr_t> = nullptr>
  void axpy_(InoutView& x,
             const InputView& y,
             const typename InoutView::value_type beta=1,
             const typename InoutView::value_type alpha=1,
             const int axis=0) {
    // Inplace: 2D + 1D -> 2D (broadcasting)
    assert(x.extent(axis) == y.extent(0));

    const auto n0 = x.extent(0), n1 = x.extent(1);
    IteratePolicy<typename InoutView::layout_type, 2> policy2d({0, 0}, {n0, n1});
    Impl::for_each(policy2d,
      [=](const int i0, const int i1) {
        const auto idx0 = (axis==0) ? i0 : i1;
        x(i0, i1) = alpha * x(i0, i1) + beta * y(idx0);
      });
  }

  template <class InoutView,
            class InputView,
            std::enable_if_t<InoutView::rank()==2 && InputView::rank()==1, std::nullptr_t> = nullptr>
  void axpy_(const InoutView& x,
             const InputView& y,
             InoutView& z,
             const typename InoutView::value_type beta=1,
             const typename InoutView::value_type alpha=1,
             const int axis=0) {
    // Outplace: 2D + 1D -> 2D (broadcasting)
    assert(x.extent(axis) == y.extent(0));
    assert(x.extents() == z.extents());

    const auto n0 = x.extent(0), n1 = x.extent(1);
    IteratePolicy<typename InoutView::layout_type, 2> policy2d({0, 0}, {n0, n1});
    Impl::for_each(policy2d,
      [=](const int i0, const int i1) {
        const auto idx0 = (axis==0) ? i0 : i1;
        z(i0, i1) = alpha * x(i0, i1) + beta * y(idx0);
      });
  }

  template <class InoutView,
            class InputView,
            std::enable_if_t<InoutView::rank()==2 && InputView::rank()==2, std::nullptr_t> = nullptr>
  void axpy_(InoutView& x,
             const InputView& y,
             const typename InoutView::value_type beta=1,
             const typename InoutView::value_type alpha=1) {
    // Inplace: 2D + 2D -> 2D
    const auto nx0 = x.extent(0), nx1 = x.extent(1);
    const auto ny0 = y.extent(0), ny1 = y.extent(1);

    if( x.extents() == y.extents() ) {
      IteratePolicy<typename InoutView::layout_type, 2> policy2d({0, 0}, {nx0, nx1});
      Impl::for_each(policy2d,
        [=](const int i0, const int i1) {
          x(i0, i1) = alpha * x(i0, i1) + beta * y(i0, i1);
        });
    } else if( ny0 == 1 && ny0 < nx0 && ny1 == nx1 ) {
      auto sub_y = submdspan(y, 0, std::experimental::full_extent);
      axpy_(x, sub_y, beta, alpha, 1);
    } else if( ny0 == nx0 && ny1 == 1 && ny1 < nx1 ) {
      auto sub_y = submdspan(y, std::experimental::full_extent, 0);
      axpy_(x, sub_y, beta, alpha, 0);
    } else if( ny0 == 1 && ny0 < nx0 && ny1 == 1 && ny1 < nx1 ) {
      IteratePolicy<typename InoutView::layout_type, 2> policy2d({0, 0}, {nx0, nx1});
      Impl::for_each(policy2d,
        [=](const int i0, const int i1) {
          x(i0, i1) = alpha * x(i0, i1) + beta * y(0, 0);
        });
    } else {
      std::runtime_error("Cannot broadcast y to x. Check the shapes of x and y.");
    }
  }

  template <class InoutView,
            class InputView,
            std::enable_if_t<InoutView::rank()==2 && InputView::rank()==2, std::nullptr_t> = nullptr>
  void axpy_(const InoutView& x,
             const InputView& y,
             InoutView& z,
             const typename InoutView::value_type beta=1,
             const typename InoutView::value_type alpha=1) {
    // Outplace: 2D + 2D -> 2D
    assert(x.extents() == z.extents());
    const auto nx0 = x.extent(0), nx1 = x.extent(1);
    const auto ny0 = y.extent(0), ny1 = y.extent(1);

    if( x.extents() == y.extents() ) {
      IteratePolicy<typename InoutView::layout_type, 2> policy2d({0, 0}, {nx0, nx1});
      Impl::for_each(policy2d,
        [=](const int i0, const int i1) {
          z(i0, i1) = alpha * x(i0, i1) + beta * y(i0, i1);
        });
    } else if( ny0 == 1 && ny0 < nx0 && ny1 == nx1 ) {
      auto sub_y = submdspan(y, 0, std::experimental::full_extent);
      axpy_(x, sub_y, z, beta, alpha, 1);
    } else if( ny0 == nx0 && ny1 == 1 && ny1 < nx1 ) {
      auto sub_y = submdspan(y, std::experimental::full_extent, 0);
      axpy_(x, sub_y, z, beta, alpha, 0);
    } else if( ny0 == 1 && ny0 < nx0 && ny1 == 1 && ny1 < nx1 ) {
      IteratePolicy<typename InoutView::layout_type, 2> policy2d({0, 0}, {nx0, nx1});
      Impl::for_each(policy2d,
        [=](const int i0, const int i1) {
          z(i0, i1) = alpha * x(i0, i1) + beta * y(0, 0);
        });
    } else {
      std::runtime_error("Cannot broadcast y to x. Check the shapes of x and y.");
    }
  }

  template <class InoutView,
            std::enable_if_t<InoutView::rank()==3, std::nullptr_t> = nullptr>
  void axpy(InoutView& x,
            const typename InoutView::value_type& y,
            const typename InoutView::value_type beta=1,
            const typename InoutView::value_type alpha=1) {
    // Inplace: 3D + 0D -> 3D (broadcasting)
    const auto n0 = x.extent(0), n1 = x.extent(1), n2 = x.extent(2);

    IteratePolicy<typename InoutView::layout_type, 3> policy3d({0, 0, 0}, {n0, n1, n2});
    Impl::for_each(policy3d,
      [=](const int i0, const int i1, const int i2) {
        x(i0, i1, i2) = alpha * x(i0, i1, i2) + beta * y;
      });
  }

  template <class InoutView,
            std::enable_if_t<InoutView::rank()==3, std::nullptr_t> = nullptr>
  void axpy(const InoutView& x,
            const typename InoutView::value_type& y,
            InoutView& z,
            const typename InoutView::value_type beta=1,
            const typename InoutView::value_type alpha=1) {
    // Outplace: 3D + 0D -> 3D (broadcasting)
    const auto n0 = x.extent(0), n1 = x.extent(1), n2 = x.extent(2);
    IteratePolicy<typename InoutView::layout_type, 3> policy3d({0, 0, 0}, {n0, n1, n2});
    Impl::for_each(policy3d,
      [=](const int i0, const int i1, const int i2) {
        z(i0, i1, i2) = alpha * x(i0, i1, i2) + beta * y;
      });
  }

  template <class InoutView,
            class InputView,
            std::enable_if_t<InoutView::rank()==3 && InputView::rank()==1, std::nullptr_t> = nullptr>
  void axpy_(InoutView& x,
             const InputView& y,
             const typename InoutView::value_type beta=1,
             const typename InoutView::value_type alpha=1,
             const int axis=0) {
    // Inplace: 3D + 1D -> 3D (broadcasting)
    assert(x.extent(axis) == y.extent(0));

    const auto n0 = x.extent(0), n1 = x.extent(1), n2 = x.extent(2);
    IteratePolicy<typename InoutView::layout_type, 3> policy3d({0, 0, 0}, {n0, n1, n2});
    Impl::for_each(policy3d,
      [=](const int i0, const int i1, const int i2) {
        const auto idx0 = (axis==0) ? i0 :
                          (axis==1) ? i1 :
                                      i2 ;
        x(i0, i1, i2) = alpha * x(i0, i1, i2) + beta * y(idx0);
      });
  }

  template <class InoutView,
            class InputView,
            std::enable_if_t<InoutView::rank()==3 && InputView::rank()==1, std::nullptr_t> = nullptr>
  void axpy_(const InoutView& x,
             const InputView& y,
             InoutView& z,
             const typename InoutView::value_type beta=1,
             const typename InoutView::value_type alpha=1,
             const int axis=0) {
    // Outplace: 3D + 1D -> 3D (broadcasting)
    assert(x.extent(axis) == y.extent(0));

    const auto n0 = x.extent(0), n1 = x.extent(1), n2 = x.extent(2);
    IteratePolicy<typename InoutView::layout_type, 3> policy3d({0, 0, 0}, {n0, n1, n2});
    Impl::for_each(policy3d,
      [=](const int i0, const int i1, const int i2) {
        const auto idx0 = (axis==0) ? i0 :
                          (axis==1) ? i1 :
                                      i2 ;
        z(i0, i1, i2) = alpha * x(i0, i1, i2) + beta * y(idx0);
      });
  }

  template <class InoutView,
            class InputView,
            std::enable_if_t<InoutView::rank()==3 && InputView::rank()==2, std::nullptr_t> = nullptr>
  void axpy_(InoutView& x,
             const InputView& y,
             const typename InoutView::value_type beta=1,
             const typename InoutView::value_type alpha=1,
             const int axis=0) {
    // Inplace: 3D + 2D -> 3D (broadcasting)
    const auto n0 = x.extent(0), n1 = x.extent(1), n2 = x.extent(2);
    IteratePolicy<typename InoutView::layout_type, 3> policy3d({0, 0, 0}, {n0, n1, n2});
    Impl::for_each(policy3d,
      [=](const int i0, const int i1, const int i2) {
        int idx0, idx1;
        if(axis==0) {
          idx0 = i1;
          idx1 = i2;
        } else if(axis==1) {
          idx0 = i0;
          idx1 = i2;
        } else {
          idx0 = i0;
          idx1 = i1;
        }
        x(i0, i1, i2) = alpha * x(i0, i1, i2) + beta * y(idx0, idx1);
      });
  }

  template <class InoutView,
            class InputView,
            std::enable_if_t<InoutView::rank()==3 && InputView::rank()==2, std::nullptr_t> = nullptr>
  void axpy_(const InoutView& x,
             const InputView& y,
             InoutView& z,
             const typename InoutView::value_type beta=1,
             const typename InoutView::value_type alpha=1,
             const int axis=0) {
    // Outplace: 3D + 2D -> 3D (broadcasting)
    const auto n0 = x.extent(0), n1 = x.extent(1), n2 = x.extent(2);
    IteratePolicy<typename InoutView::layout_type, 3> policy3d({0, 0, 0}, {n0, n1, n2});
    Impl::for_each(policy3d,
      [=](const int i0, const int i1, const int i2) {
        int idx0, idx1;
        if(axis==0) {
          idx0 = i1;
          idx1 = i2;
        } else if(axis==1) {
          idx0 = i0;
          idx1 = i2;
        } else {
          idx0 = i0;
          idx1 = i1;
        }
        z(i0, i1, i2) = alpha * x(i0, i1, i2) + beta * y(idx0, idx1);
      });
  }

  template <class InoutView,
            class InputView,
            std::enable_if_t<InoutView::rank()==3 && InputView::rank()==3, std::nullptr_t> = nullptr>
  void axpy_(InoutView& x, 
             const InputView& y,
             typename InoutView::value_type beta=1,
             typename InoutView::value_type alpha=1) {
    // Inplace: 3D + 3D -> 3D
    const auto nx0 = x.extent(0), nx1 = x.extent(1), nx2 = x.extent(2);
    const auto ny0 = y.extent(0), ny1 = y.extent(1), ny2 = y.extent(2);

    if( x.extents() == y.extents() ) {
      IteratePolicy<typename InoutView::layout_type, 3> policy3d({0, 0, 0}, {nx0, nx1, nx2});
      Impl::for_each(policy3d,
        [=](const int i0, const int i1, const int i2) {
          x(i0, i1, i2) = alpha * x(i0, i1, i2) + beta * y(i0, i1, i2);
        });
    } else if( ny0 == 1 && ny0 < nx0 && ny1 == nx1 && ny2 == nx2 ) {
      auto sub_y = submdspan(y, 0, std::experimental::full_extent, std::experimental::full_extent);
      axpy_(x, sub_y, beta, alpha, 0);
    } else if( ny0 == 1 && ny0 < nx0 && ny1 == 1 && ny1 < nx1 && ny2 == nx2 ) {
      auto sub_y = submdspan(y, 0, 0, std::experimental::full_extent);
      axpy_(x, sub_y, beta, alpha, 2);
    } else if( ny0 == 1 && ny0 < nx0 && ny1 == nx1 && ny2 == 1 && ny2 < nx2 ) {
      auto sub_y = submdspan(y, 0, std::experimental::full_extent, 0);
      axpy_(x, sub_y, beta, alpha, 1);
    } else if( ny0 == nx0 && ny1 == 1 && ny1 < nx1 && ny2 == nx2 ) {
      auto sub_y = submdspan(y, std::experimental::full_extent, 0, std::experimental::full_extent);
      axpy_(x, sub_y, beta, alpha, 1);
    } else if( ny0 == nx0 && ny1 == 1 && ny1 < nx1 && ny2 == 1 && ny2 < nx2 ) {
      auto sub_y = submdspan(y, std::experimental::full_extent, 0, 0);
      axpy_(x, sub_y, beta, alpha, 0);
    } else if( ny0 == nx0 && ny1 == nx1 && ny2 == 1 && ny2 < nx2 ) {
      auto sub_y = submdspan(y, std::experimental::full_extent, std::experimental::full_extent, 0);
      axpy_(x, sub_y, beta, alpha, 2);
    } else if( ny0 == 1 && ny0 < nx0 && ny1 == 1 && ny1 < nx1 && ny2 == 1 && ny2 < nx2 ) {
      IteratePolicy<typename InoutView::layout_type, 3> policy3d({0, 0, 0}, {nx0, nx1, nx2});
      Impl::for_each(policy3d,
        [=](const int i0, const int i1, const int i2) {
          x(i0, i1, i2) = alpha * x(i0, i1, i2) + beta * y(0, 0, 0);
        });
    } else {
      std::runtime_error("Cannot broadcast y to x. Check the shapes of x and y.");
    }
  }

  template <class InoutView,
            class InputView,
            std::enable_if_t<InoutView::rank()==3 && InputView::rank()==3, std::nullptr_t> = nullptr>
  void axpy_(const InoutView& x,
             const InputView& y,
             InoutView& z,
             typename InoutView::value_type beta=1,
             typename InoutView::value_type alpha=1) {
    // Outplace: 3D + 3D -> 3D
    const auto nx0 = x.extent(0), nx1 = x.extent(1), nx2 = x.extent(2);
    const auto ny0 = y.extent(0), ny1 = y.extent(1), ny2 = y.extent(2);

    if( x.extents() == y.extents() ) {
      IteratePolicy<typename InoutView::layout_type, 3> policy3d({0, 0, 0}, {nx0, nx1, nx2});
      Impl::for_each(policy3d,
        [=](const int i0, const int i1, const int i2) {
          z(i0, i1, i2) = alpha * x(i0, i1, i2) + beta * y(i0, i1, i2);
        });
    } else if( ny0 == 1 && ny0 < nx0 && ny1 == nx1 && ny2 == nx2 ) {
      auto sub_y = submdspan(y, 0, std::experimental::full_extent, std::experimental::full_extent);
      axpy_(x, sub_y, z, beta, alpha, 0);
    } else if( ny0 == 1 && ny0 < nx0 && ny1 == 1 && ny1 < nx1 && ny2 == nx2 ) {
      auto sub_y = submdspan(y, 0, 0, std::experimental::full_extent);
      axpy_(x, sub_y, z, beta, alpha, 2);
    } else if( ny0 == 1 && ny0 < nx0 && ny1 == nx1 && ny2 == 1 && ny2 < nx2 ) {
      auto sub_y = submdspan(y, 0, std::experimental::full_extent, 0);
      axpy_(x, sub_y, z, beta, alpha, 1);
    } else if( ny0 == nx0 && ny1 == 1 && ny1 < nx1 && ny2 == nx2 ) {
      auto sub_y = submdspan(y, std::experimental::full_extent, 0, std::experimental::full_extent);
      axpy_(x, sub_y, z, beta, alpha, 1);
    } else if( ny0 == nx0 && ny1 == 1 && ny1 < nx1 && ny2 == 1 && ny2 < nx2 ) {
      auto sub_y = submdspan(y, std::experimental::full_extent, 0, 0);
      axpy_(x, sub_y, z, beta, alpha, 0);
    } else if( ny0 == nx0 && ny1 == nx1 && ny2 == 1 && ny2 < nx2 ) {
      auto sub_y = submdspan(y, std::experimental::full_extent, std::experimental::full_extent, 0);
      axpy_(x, sub_y, z, beta, alpha, 2);
    } else if( ny0 == 1 && ny0 < nx0 && ny1 == 1 && ny1 < nx1 && ny2 == 1 && ny2 < nx2 ) {
      IteratePolicy<typename InoutView::layout_type, 3> policy3d({0, 0, 0}, {nx0, nx1, nx2});
      Impl::for_each(policy3d,
        [=](const int i0, const int i1, const int i2) {
          z(i0, i1, i2) = alpha * x(i0, i1, i2) + beta * y(0, 0, 0);
        });
    } else {
      std::runtime_error("Cannot broadcast y to x. Check the shapes of x and y.");
    }
  }

  template <class InoutView,
            class InputView,
            std::enable_if_t<InoutView::rank()>=1 && InputView::rank()>=1, std::nullptr_t> = nullptr>
  void axpy(InoutView& x, const InputView& y, typename InoutView::value_type beta=1, typename InoutView::value_type alpha=1) {
    /* Compute y = alpha * x + beta * y (In-place) */
    static_assert( std::is_same_v<typename InoutView::value_type, typename InputView::value_type> );
    static_assert( std::is_same_v<typename InoutView::layout_type, typename InputView::layout_type> );
    static_assert( InoutView::rank() >= InputView::rank() );
    static_assert( InoutView::rank() <= 3 );

    axpy_(x, y, beta, alpha);
  }

  template <class InoutView,
            class InputView,
            std::enable_if_t<InoutView::rank()>=1 && InputView::rank()>=1, std::nullptr_t> = nullptr>
  void axpy(const InoutView& x, const InputView& y, InoutView& z, typename InoutView::value_type beta=1, typename InoutView::value_type alpha=1) {
    /* Compute z = alpha * x + beta * y (Out-place) */
    static_assert( std::is_same_v<typename InoutView::value_type, typename InputView::value_type> );
    static_assert( std::is_same_v<typename InoutView::layout_type, typename InputView::layout_type> );
    static_assert( InoutView::rank() >= InputView::rank() );
    static_assert( InoutView::rank() <= 3 );
    assert( x.extents() == z.extents() );

    axpy_(x, y, z, beta, alpha);
  }

  /* zeros_like, ones_like, identity, diag */
  template <class InoutView>
  void zeros_like(InoutView& a) {
    using value_type = InoutView::value_type;
    value_type* ptr_a = a.data_handle();
    const auto n = a.size();

    // Not quite sure, this is a better strategy
    IteratePolicy<typename InoutView::layout_type, 1> policy1d(n);
    Impl::for_each(policy1d,
      [=](const int idx) {
        ptr_a[idx] = static_cast<value_type>(0);
    });
  }

  template <class InputView, class OutputView>
  void zeros_like(const InputView& in, OutputView& out) {
    static_assert( std::is_same_v<typename InputView::value_type, typename InputView::value_type> );
    static_assert( std::is_same_v<typename InputView::layout_type, typename InputView::layout_type> );
    assert(in.extents() == out.extents());
    using value_type = InputView::value_type;
    value_type* ptr_out = out.data_handle();
    const auto n = in.size();

    IteratePolicy<typename InputView::layout_type, 1> policy1d(n);
    Impl::for_each(policy1d,
      [=](const int idx) {
        ptr_out[idx] = static_cast<value_type>(0);
    });
  }

  template <class InoutView>
  void ones_like(InoutView& a) {
    using value_type = InoutView::value_type;
    value_type* ptr_a = a.data_handle();
    const auto n = a.size();
    IteratePolicy<typename InoutView::layout_type, 1> policy1d(n);
    Impl::for_each(policy1d,
      [=](const int idx) {
        ptr_a[idx] = static_cast<value_type>(1);
    });
  }

  template <class InputView, class OutputView>
  void ones_like(const InputView& in, OutputView& out) {
    static_assert( std::is_same_v<typename InputView::value_type, typename InputView::value_type> );
    static_assert( std::is_same_v<typename InputView::layout_type, typename InputView::layout_type> );
    assert(in.extents() == out.extents());
    using value_type = InputView::value_type;
    value_type* ptr_out = out.data_handle();
    const auto n = in.size();
    IteratePolicy<typename InputView::layout_type, 1> policy1d(n);
    Impl::for_each(policy1d,
      [=](const int idx) {
        ptr_out[idx] = static_cast<value_type>(1);
    });
  }

  /* Construct diagonal matrix for vector with arbitral exponent */
  template <class MatrixView,
            std::enable_if_t<MatrixView::rank()==2, std::nullptr_t> = nullptr>
  void identity(MatrixView& out) {
    assert( out.extent(0) == out.extent(1) ); // Square matrix
    using value_type = MatrixView::value_type;

    zeros_like(out);
    const auto n = out.extent(0);

    IteratePolicy<typename MatrixView::layout_type, 1> policy1d(n);
    Impl::for_each(policy1d,
      [=](const int idx) {
        out(idx, idx) = static_cast<value_type>(1);
    });
  }

  template <class MatrixView,
            std::enable_if_t<MatrixView::rank()==3, std::nullptr_t> = nullptr>
  void identity(MatrixView& out) {
    assert( out.extent(0) == out.extent(1) ); // Square matrix
    using value_type = MatrixView::value_type;

    zeros_like(out);
    const auto n0 = out.extent(0), n2 = out.extent(2);

    IteratePolicy<typename MatrixView::layout_type, 2> policy2d({0, 0}, {n0, n2});
    Impl::for_each(policy2d,
      [=](const int ix, const int iz) {
        out(ix, ix, iz) = static_cast<value_type>(1);
    });
  }

  /* Construct diagonal matrix for vector with arbitral exponent */
  template <class MatrixView, class VectorView,
            std::enable_if_t<MatrixView::rank()==2 && VectorView::rank()==1, std::nullptr_t> = nullptr>
  void diag(const VectorView& v, MatrixView& out, typename MatrixView::value_type exponent=1) {
    static_assert( std::is_same_v<typename MatrixView::value_type, typename VectorView::value_type> );
    static_assert( std::is_same_v<typename MatrixView::layout_type, typename VectorView::layout_type> );
    assert( v.extent(0) == out.extent(0) );
    assert( out.extent(0) == out.extent(1) ); // Square matrix

    zeros_like(out);
    const auto n = v.extent(0);
    IteratePolicy<typename MatrixView::layout_type, 1> policy1d(n);
    Impl::for_each(policy1d,
      [=](const int idx) {
        out(idx, idx) = pow(v(idx), exponent);
    });
  }

  template <class MatrixView, class VectorView,
            std::enable_if_t<MatrixView::rank()==3 && VectorView::rank()==2, std::nullptr_t> = nullptr>
  void diag(const VectorView& v, MatrixView& out, typename MatrixView::value_type exponent=1) {
    static_assert( std::is_same_v<typename MatrixView::value_type, typename VectorView::value_type> );
    static_assert( std::is_same_v<typename MatrixView::layout_type, typename VectorView::layout_type> );
    assert( v.extent(0) == out.extent(0) );
    assert( out.extent(0) == out.extent(1) ); // Square matrix
    assert( out.extent(2) == v.extent(1) ); // batch size

    zeros_like(out);
    const auto n0 = out.extent(0), n2 = out.extent(2);
    IteratePolicy<typename MatrixView::layout_type, 2> policy2d({0, 0}, {n0, n2});
    Impl::for_each(policy2d,
      [=](const int ix, const int iz) {
        out(ix, ix, iz) = pow(v(ix, iz), exponent);
    });
  }

  template <class ViewType>
  auto squeeze(const ViewType& x, int axis=-1) {
    using value_type = ViewType::value_type;
    using size_type = ViewType::size_type;
    using layout_type = ViewType::layout_type;
    int reduce_dim = axis==-1 ? ViewType::rank()-1 : axis;
    assert(reduce_dim < ViewType::rank());
    assert(x.extent(reduce_dim) == 1);

    const std::size_t rank = ViewType::rank();

    using mdspan_type = stdex::mdspan<value_type, stdex::dextents<size_type, rank-1>, layout_type>;
    using extent_type = std::array<std::size_t, rank-1>;

    std::vector<std::size_t> v;
    for(int i=0; i<rank; i++) {
      if(i==reduce_dim) continue;
      v.push_back(x.extent(i));
    }
    extent_type out_shape;
    std::move(v.begin(), v.end(), out_shape.begin());

    return mdspan_type( x.data_handle(), out_shape );
  }

  template <class ViewType, std::size_t N>
  auto reshape(const ViewType& x, const std::array<std::size_t, N>& shape) {
    using value_type = ViewType::value_type;
    using size_type = ViewType::size_type;
    using layout_type = ViewType::layout_type;

    const std::size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    assert( size == x.size() );

    using mdspan_type = stdex::mdspan<value_type, stdex::dextents<size_type, N>, layout_type>;

    return mdspan_type( x.data_handle(), shape );
  }

  template <class ViewType>
  void deep_copy(const ViewType& in, ViewType& out) {
    assert(in.extents() == out.extents());
    const auto n = in.size();
    thrust::copy(thrust::device, in.data_handle(), in.data_handle()+n, out.data_handle());
  }

};

#endif
