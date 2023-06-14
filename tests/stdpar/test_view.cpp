#include <gtest/gtest.h>
#include "Types.hpp"
#include <stdpar/Parallel_For.hpp>

void allocate_inside_function(RealView2D& reference_to_a_View, const int n, const int m) {
  // Set values on device via move assign
  reference_to_a_View = RealView2D("simple", n, m);

  Iterate_policy<2> policy2d({0, 0}, {n, m});
  auto _reference_to_a_View = reference_to_a_View.mdspan();
  Impl::for_each(policy2d,
    [=] (const int i, const int j){
      _reference_to_a_View(i, j) = i + j * 0.2 + 0.01;
    });
}

void set_inside_function(RealView2D shallow_copy_to_a_View) {
  // Set values on device via shallow copy
  const int n = shallow_copy_to_a_View.extent(0);
  const int m = shallow_copy_to_a_View.extent(1);

  Iterate_policy<2> policy2d({0, 0}, {n, m});

  auto _shallow_copy_to_a_View = shallow_copy_to_a_View.mdspan();
  Impl::for_each(policy2d,
    [=] (const int i, const int j){
      _shallow_copy_to_a_View(i, j) = i + j * 0.2 + 0.01;
    });
}

void test_copy_constructor() {
  RealView2D simple("simple", 16, 16);
  RealView2D reference("reference", 16, 16);

  set_inside_function(simple);

  // Set in the main function
  const int n = reference.extent(0);
  const int m = reference.extent(1);

  Iterate_policy<2> policy2d({0, 0}, {n, m});

  auto _reference = reference.mdspan();
  Impl::for_each(policy2d,
    [=] (const int i, const int j){
      _reference(i, j) = i + j * 0.2 + 0.01;
    });

  simple.updateSelf();
  reference.updateSelf();

  // Check if the host data are identical
  ASSERT_EQ( simple.size(), reference.size() );
  for(int j=0; j<m; j++) {
    for(int i=0; i<n; i++) {
      ASSERT_EQ( simple(i, j), reference(i, j) );
      ASSERT_NE( simple(i, j), 0.0 ); // Just to make sure simple has some value
    }
  }
}

void test_assignment_operator() {
  // [NOTE] Do not recommend to use assignement opertor
  RealView2D simple;
  RealView2D assinged_via_simple("assinged_via_simple", 16, 16);
  RealView2D reference("reference", 16, 16);

  simple = assinged_via_simple;

  // Set in the main function
  const int n = reference.extent(0);
  const int m = reference.extent(1);

  Iterate_policy<2> policy2d({0, 0}, {n, m});
  auto _reference = reference.mdspan();
  auto _simple = simple.mdspan();
  Impl::for_each(policy2d,
    [=] (const int i, const int j){
      _reference(i, j) = i + j * 0.2 + 0.01;
      _simple(i, j) = i + j * 0.2 + 0.01;
    });

  reference.updateSelf();
  assinged_via_simple.updateSelf();

  // Check if the host data are identical
  ASSERT_EQ( assinged_via_simple.size(), reference.size() );
  for(int j=0; j<m; j++) {
    for(int i=0; i<n; i++) {
      ASSERT_EQ( assinged_via_simple(i, j), reference(i, j) );
      ASSERT_NE( assinged_via_simple(i, j), 0.0 ); // Just to make sure simple has some value
    }
  }
}

void test_move_constructor() {
  RealView2D simple;
  RealView2D moved_reference("reference", 16, 16);
  RealView2D reference("reference", 16, 16);

  // Set in the main function
  const int n = moved_reference.extent(0);
  const int m = moved_reference.extent(1);

  Iterate_policy<2> policy2d({0, 0}, {n, m});

  auto _reference = reference.mdspan();
  auto _moved_reference = moved_reference.mdspan();
  Impl::for_each(policy2d,
    [=] (const int i, const int j){
      _moved_reference(i, j) = i + j * 0.2 + 0.01;
      _reference(i, j) = _moved_reference(i, j);
    });

  reference.updateSelf();
  moved_reference.updateSelf();
  simple = std::move(moved_reference);

  // Check if the host data are identical
  ASSERT_EQ( simple.size(), reference.size() );
  for(int j=0; j<m; j++) {
    for(int i=0; i<n; i++) {
      ASSERT_EQ( simple(i, j), reference(i, j) );
      ASSERT_NE( simple(i, j), 0.0 ); // Just to make sure simple has some value
    }
  }

  simple.updateSelf();
  // Check if the device data are identical
  for(int j=0; j<m; j++) {
    for(int i=0; i<n; i++) {
      ASSERT_EQ( simple(i, j), reference(i, j) );
      ASSERT_NE( simple(i, j), 0.0 ); // Just to make sure simple has some value
    }
  }

}

void test_move_assignment_operator() {
  RealView2D simple;
  RealView2D reference("reference", 16, 16);

  // Set in the main function
  const int n = reference.extent(0);
  const int m = reference.extent(1);

  allocate_inside_function(simple, n, m);

  Iterate_policy<2> policy2d({0, 0}, {n, m});

  auto _reference = reference.mdspan();
  Impl::for_each(policy2d,
    [=] (const int i, const int j){
      _reference(i, j) = i + j * 0.2 + 0.01;
    });

  simple.updateSelf();
  reference.updateSelf();

  // Check if the host data are identical
  ASSERT_EQ( simple.size(), reference.size() );
  for(int j=0; j<m; j++) {
    for(int i=0; i<n; i++) {
      ASSERT_EQ( simple(i, j), reference(i, j) );
      ASSERT_NE( simple(i, j), 0.0 ); // Just to make sure simple has some value
    }
  }
}

void test_swap() {
  /* Swap a and b
   */
  RealView2D a("a", 16, 16), b("b", 16, 16);
  RealView2D a_ref("b", 16, 16), b_ref("a", 16, 16);

  // Set in the main function
  const int n = a.extent(0);
  const int m = a.extent(1);

  Iterate_policy<2> policy2d({0, 0}, {n, m});

  auto _a = a.mdspan();
  auto _b = b.mdspan();
  auto _a_ref = a_ref.mdspan();
  auto _b_ref = b_ref.mdspan();

  Impl::for_each(policy2d,
    [=] (const int i, const int j){
      _a(i, j) = i + j * 0.2 + 0.01;
      _b(i, j) = i *0.3 + j * 0.5 + 0.01;

      _a_ref(i, j) = _b(i, j);
      _b_ref(i, j) = _a(i, j);
    });

  a.updateSelf();
  b.updateSelf();
  a_ref.updateSelf();
  b_ref.updateSelf();

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
      ASSERT_EQ( a(i, j), a_ref(i, j) );
      ASSERT_EQ( b(i, j), b_ref(i, j) );
      ASSERT_NE( a(i, j), 0.0 );
      ASSERT_NE( b(i, j), 0.0 );
    }
  }

  // check device data are identical
  a.updateSelf();
  b.updateSelf();
  for(int j=0; j<m; j++) {
    for(int i=0; i<n; i++) {
      ASSERT_EQ( a(i, j), a_ref(i, j) );
      ASSERT_EQ( b(i, j), b_ref(i, j) );
      ASSERT_NE( a(i, j), 0.0 );
      ASSERT_NE( b(i, j), 0.0 );
    }
  }
}

TEST( STDPAR_VIEW, DEFAULT_CONSTRUCTOR ) {
  RealView2D empty; 
  RealView2D simple("simple", std::array<size_type, 2>{2, 3}); // Simple constructor
  RealView2D Kokkos_like("kokkos_like", 2, 3); // Kokkos like constructor
}

TEST( STDPAR_VIEW, COPY_CONSTRUCTOR ) {
  test_copy_constructor();
}

TEST( STDPAR_VIEW, ASSIGN ) {
  test_assignment_operator();
}

TEST( STDPAR_VIEW, MOVE ) {
  test_move_constructor();
}

TEST( STDPAR_VIEW, MOVE_ASSIGN ) {
  test_move_assignment_operator();
}

TEST( STDPAR_VIEW, SWAP ) {
  test_swap();
}
