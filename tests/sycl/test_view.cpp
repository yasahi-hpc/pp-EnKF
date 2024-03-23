#include <gtest/gtest.h>
#include "Types.hpp"
#include "Test_Helper.hpp"

class TestViews : public ::testing::Test {
protected:
  std::unique_ptr<sycl::queue> queue_;

  virtual void SetUp() {
    auto selector = sycl::gpu_selector_v;
    try {
      queue_ = std::make_unique<sycl::queue>(selector, exception_handler, sycl::property_list{sycl::property::queue::in_order{}});
      queue_->wait();
    } catch (std::exception const& e) {
      std::cout << "An exception is caught intializing a queue.\n";
      std::terminate();
    }
  }
};

void allocate_inside_function(sycl::queue& q, RealView2D& reference_to_a_View, const int n, const int m) {
  // Set values on device via move assign
  reference_to_a_View = RealView2D(q, "simple", n, m);

  auto _reference_to_a_View = reference_to_a_View.mdspan();

  // Sycl range
  sycl::range<2> global_range(n, m);

  // Get the local range for the 3D parallel for loop
  sycl::range<2> local_range = sycl::range<2>(4, 4);

  // Create a 3D nd_range using the global and local ranges
  sycl::nd_range<2> nd_range(global_range, local_range);

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
      nd_range,
      [=](sycl::nd_item<2> item) {
        const int i = item.get_global_id(0);
        const int j = item.get_global_id(1);
        _reference_to_a_View(i, j) = i + j * 0.2 + 0.01;
      });
    });
  q.wait();
}

void set_inside_function(sycl::queue& q, RealView2D shallow_copy_to_a_View) {
  // Set values on device via shallow copy
  const int n = shallow_copy_to_a_View.extent(0);
  const int m = shallow_copy_to_a_View.extent(1);

  auto _shallow_copy_to_a_View = shallow_copy_to_a_View.mdspan();

  // Sycl range
  sycl::range<2> global_range(n, m);

  // Get the local range for the 3D parallel for loop
  sycl::range<2> local_range = sycl::range<2>(4, 4);

  // Create a 3D nd_range using the global and local ranges
  sycl::nd_range<2> nd_range(global_range, local_range);

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
      nd_range,
      [=](sycl::nd_item<2> item) {
        const int i = item.get_global_id(0);
        const int j = item.get_global_id(1);
        _shallow_copy_to_a_View(i, j) = i + j * 0.2 + 0.01;
      });
    });
  q.wait();
}

void test_copy_constructor(sycl::queue& q) {
  RealView2D simple(q, "simple", 16, 16);
  RealView2D reference(q, "reference", 16, 16);

  set_inside_function(q, simple);

  // Set in the main function
  const int n = reference.extent(0);
  const int m = reference.extent(1);

  auto _reference = reference.mdspan();
  for(int j=0; j<m; j++) {
    for(int i=0; i<n; i++) {
      _reference(i, j) = i + j * 0.2 + 0.01;
    }
  }

  // Check if the host data are identical
  ASSERT_EQ( simple.size(), reference.size() );
  for(int j=0; j<m; j++) {
    for(int i=0; i<n; i++) {
      EXPECT_DOUBLE_EQ( simple(i, j), reference(i, j) );
      EXPECT_NE( simple(i, j), 0.0 ); // Just to make sure simple has some value
    }
  }
}

void test_move_constructor(sycl::queue& q) {
  RealView2D simple;
  RealView2D moved_reference(q, "reference", 16, 16);
  RealView2D reference(q, "reference", 16, 16);

  // Set in the main function
  const int n = moved_reference.extent(0);
  const int m = moved_reference.extent(1);

  auto _reference = reference.mdspan();
  for(int j=0; j<m; j++) {
    for(int i=0; i<n; i++) {
      _reference(i, j) = i + j * 0.2 + 0.01;
    }
  }

  set_inside_function(q, moved_reference);

  // simple is set by move
  simple = std::move(moved_reference);

  // Check if the host data are identical
  ASSERT_EQ( simple.size(), simple.size() );
  for(int j=0; j<m; j++) {
    for(int i=0; i<n; i++) {
      EXPECT_DOUBLE_EQ( simple(i, j), reference(i, j) );
      EXPECT_NE( simple(i, j), 0.0 ); // Just to make sure simple has some value
    }
  }
}

void test_assignment_operator(sycl::queue& q) {
  // [NOTE] Do not recommend to use assignement opertor
  RealView2D simple;
  RealView2D assinged_via_simple(q, "assinged_via_simple", 16, 16);
  RealView2D reference(q, "reference", 16, 16);

  simple = assinged_via_simple;
  set_inside_function(q, simple);

  // Set in the main function
  const int n = reference.extent(0);
  const int m = reference.extent(1);

  auto _reference = reference.mdspan();
  for(int j=0; j<m; j++) {
    for(int i=0; i<n; i++) {
      _reference(i, j) = i + j * 0.2 + 0.01;
    }
  }

  // Check if the host data are identical
  ASSERT_EQ( assinged_via_simple.size(), reference.size() );
  for(int j=0; j<m; j++) {
    for(int i=0; i<n; i++) {
      EXPECT_DOUBLE_EQ( assinged_via_simple(i, j), reference(i, j) );
      EXPECT_NE( assinged_via_simple(i, j), 0.0 ); // Just to make sure simple has some value
    }
  }
}

void test_move_assignment_operator(sycl::queue& q) {
  RealView2D simple;
  RealView2D reference(q, "reference", 16, 16);

  // Set in the main function
  const int n = reference.extent(0);
  const int m = reference.extent(1);

  allocate_inside_function(q, simple, n, m);

  auto _reference = reference.mdspan();
  for(int j=0; j<m; j++) {
    for(int i=0; i<n; i++) {
      _reference(i, j) = i + j * 0.2 + 0.01;
    }
  }

  // Check if the host data are identical
  ASSERT_EQ( simple.size(), reference.size() );
  for(int j=0; j<m; j++) {
    for(int i=0; i<n; i++) {
      EXPECT_DOUBLE_EQ( simple(i, j), reference(i, j) );
      EXPECT_NE( simple(i, j), 0.0 ); // Just to make sure simple has some value
    }
  }
}

void test_swap(sycl::queue& q) {
  /* Swap a and b
   */
  RealView2D a(q, "a", 16, 16), b(q, "b", 16, 16);
  RealView2D a_ref(q, "b", 16, 16), b_ref(q, "a", 16, 16);

  // Set in the main function
  const int n = a.extent(0);
  const int m = a.extent(1);

  auto _a = a.mdspan();
  auto _b = b.mdspan();
  auto _a_ref = a_ref.mdspan();
  auto _b_ref = b_ref.mdspan();

  for(int j=0; j<m; j++) {
    for(int i=0; i<n; i++) {
      _a(i, j) = i + j * 0.2 + 0.01;
      _b(i, j) = i *0.3 + j * 0.5 + 0.01;

      _a_ref(i, j) = _b(i, j);
      _b_ref(i, j) = _a(i, j);
    }
  }

  // Swap a and b
  a.swap(b);

  // Check meta data are identical
  ASSERT_EQ( a.size(), a_ref.size() );
  ASSERT_EQ( b.size(), b_ref.size() );
  ASSERT_EQ( a.name(), a_ref.name() );
  ASSERT_EQ( b.name(), b_ref.name() );

  // check host data are correctly swapped
  for(int j=0; j<m; j++) {
    for(int i=0; i<n; i++) {
      EXPECT_DOUBLE_EQ( a(i, j), a_ref(i, j) );
      EXPECT_DOUBLE_EQ( b(i, j), b_ref(i, j) );
      EXPECT_NE( a(i, j), 0.0 );
      EXPECT_NE( b(i, j), 0.0 );
    }
  }
}

TEST_F( TestViews, DEFAULT_CONSTRUCTOR ) {
  RealView2D empty;
  RealView2D simple(*queue_, "simple", std::array<size_type, 2>{2, 3}); // Simple constructor
  RealView2D Kokkos_like(*queue_, "kokkos_like", 2, 3); // Kokkos like constructor
}

TEST_F( TestViews, COPY_CONSTRUCTOR ) {
  test_copy_constructor(*queue_);
}

TEST_F( TestViews, ASSIGN ) {
  test_assignment_operator(*queue_);
}

TEST_F( TestViews, MOVE ) {
  test_move_constructor(*queue_);
}

TEST_F( TestViews, MOVE_ASSIGN ) {
  test_move_assignment_operator(*queue_);
}

TEST_F( TestViews, SWAP ) {
  test_swap(*queue_);
}