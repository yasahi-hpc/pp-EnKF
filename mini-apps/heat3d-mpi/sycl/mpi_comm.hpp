#ifndef MPI_COMM_HPP
#define MPI_COMM_HPP

#include <cassert>
#include <vector>
#include <complex>
#include <mpi.h>
#include <memory>
#include <sycl/sycl.hpp>
#include "../types.hpp"
#include "../utils.hpp"
#include "../timer.hpp"
#include "functors.hpp"

constexpr int UP     = 0;
constexpr int DOWN   = 1;
constexpr int LEFT   = 2;
constexpr int RIGHT  = 3;
constexpr int TOP    = 4;
constexpr int BOTTOM = 5;

template <typename RealType> using Complex = std::complex<RealType>;

template <typename T,
          std::enable_if_t<std::is_same_v<T, int             > ||
                           std::is_same_v<T, float           > ||
                           std::is_same_v<T, double          > ||
                           std::is_same_v<T, Complex<float>  > ||
                           std::is_same_v<T, Complex<double> >
                           , std::nullptr_t> = nullptr
>
MPI_Datatype get_mpi_data_type() {
  MPI_Datatype type;
  if(std::is_same_v<T, int             >) type = MPI_INT;
  if(std::is_same_v<T, float           >) type = MPI_FLOAT;
  if(std::is_same_v<T, double          >) type = MPI_DOUBLE;
  if(std::is_same_v<T, Complex<float>  >) type = MPI_COMPLEX;
  if(std::is_same_v<T, Complex<double> >) type = MPI_DOUBLE_COMPLEX;

  return type;
}

template <typename RealType>
struct Halo {
  using RealView2D = View2D<RealType>;
  using Shpae2D = shape_type<2>;

private:
  sycl::queue q_;
  RealType *left_, *right_;
  Shpae2D extents_;

  std::string name_;
  std::size_t size_;
  int left_rank_, right_rank_;
  int left_tag_, right_tag_;
  MPI_Comm communicator_;
  MPI_Datatype mpi_data_type_;
  bool is_comm_;

public:
  Halo() = delete;
  Halo(sycl::queue& q, const std::string name, Shpae2D shape, int left_rank, int right_rank,
       int left_tag, int right_tag, MPI_Comm communicator, bool is_comm)
    : q_(q), name_(name), extents_(shape), left_rank_(left_rank), right_rank_(right_rank),
    left_tag_(left_tag), right_tag_(right_tag), communicator_(communicator), is_comm_(is_comm) {

    size_ = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<std::size_t>());

    left_  = sycl::malloc_shared<RealType>(size_, q_);
    right_ = sycl::malloc_shared<RealType>(size_, q_);

    mpi_data_type_ = get_mpi_data_type<RealType>();
  }

  Halo(Halo&& rhs) noexcept
    : q_(rhs.q_), name_(rhs.name_), extents_(rhs.extents_), size_(rhs.size_), left_rank_(rhs.left_rank_), right_rank_(rhs.right_rank_),
    left_(rhs.left_), right_(rhs.right_), left_tag_(rhs.left_tag_), right_tag_(rhs.right_tag_),
    communicator_(rhs.communicator_), mpi_data_type_(rhs.mpi_data_type_), is_comm_(rhs.is_comm_) {
  }

  ~Halo() {
    sycl::free(left_,  q_);
    sycl::free(right_, q_);
  }

  // Getters
  RealView2D left() { return RealView2D(left_, extents_); }
  RealView2D right() { return RealView2D(right_, extents_); }

  RealType*& left_ptr() {return left_;}
  RealType*& right_ptr() {return right_;}

  std::size_t size() const { return size_; }
  int left_rank() const { return left_rank_; }
  int right_rank() const { return right_rank_; }
  int left_tag() const { return left_tag_; }
  int right_tag() const { return right_tag_; }
  MPI_Comm communicator() const { return communicator_; }
  MPI_Datatype type() const { return mpi_data_type_; }
  bool is_comm() const {return is_comm_; }
};

class Comm {
  // ID of the MPI process
  int rank_;

  // Number of MPI processes
  int size_;

  // Data shape
  std::vector<std::size_t> shape_;

  // MPI topology
  std::vector<int> topology_;
  std::vector<int> cart_rank_;

  // Halos
  using HaloType = Halo<double>;
  std::unique_ptr< HaloType > x_send_halo_;
  std::unique_ptr< HaloType > x_recv_halo_;
  std::unique_ptr< HaloType > y_send_halo_;
  std::unique_ptr< HaloType > y_recv_halo_;
  std::unique_ptr< HaloType > z_send_halo_;
  std::unique_ptr< HaloType > z_recv_halo_;

  int halo_width_;
  sycl::queue q_;

public:
  Comm() = delete;
  Comm(int& argc,
       char** argv,
       sycl::queue& q,
       const std::vector<std::size_t>& shape,
       const std::vector<int>& topology)
    : q_(q), shape_(shape), topology_(topology), halo_width_(1) {
    int required = MPI_THREAD_MULTIPLE;
    int provided;
    ::MPI_Init_thread(&argc, &argv, required, &provided);
    ::MPI_Comm_size(MPI_COMM_WORLD, &size_);
    ::MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

    // setDevice( rank_ );
    setTopology();
  }
  ~Comm() {}

  void finalize() { ::MPI_Finalize(); }
  bool is_master() { return rank_==0; }
  int size() const { return size_; }
  int rank() const { return rank_; }
  int cart_rank(size_t i) const {
    assert( i < cart_rank_.size() );
    return cart_rank_.at(i);
  }
  std::vector<int> cart_rank() const {
    return cart_rank_;
  }
  int topology(size_t i) const {
    assert( i < topology_.size() );
    return topology_.at(i);
  }
  std::vector<int> topology() const {
    return topology_;
  }

  std::unique_ptr< HaloType >& send_buffer(std::size_t i) {
    if(i == 0) {
      return x_send_halo_;
    } if(i == 1) {
      return y_send_halo_;
    } else {
      return z_send_halo_;
    }
  }

  std::unique_ptr< HaloType >& recv_buffer(std::size_t i) {
    if(i == 0) {
      return x_recv_halo_;
    } if(i == 1) {
      return y_recv_halo_;
    } else {
      return z_recv_halo_;
    }
  }

private:
  void setTopology() {
    // Check the topology size
    assert(topology_.size() == 3);
    int topology_size = std::accumulate(topology_.begin(), topology_.end(), 1, std::multiplies<int>());
    assert(topology_size == size_);

    // Create a Cartesian Communicator
    constexpr int ndims = 3;
    int periods[ndims] = {1, 1, 1}; // Periodic in all directions
    int reorder = 1;
    int old_rank = rank_;
    MPI_Comm cart_comm;

    ::MPI_Cart_create(MPI_COMM_WORLD, ndims, topology_.data(), periods, reorder, &cart_comm);
    ::MPI_Comm_rank(cart_comm, &rank_);

    if(rank_ != old_rank) {
      std::cout << "Rank change: from " << old_rank << " to " << rank_ << std::endl;
    }

    // Define new coordinate
    cart_rank_.resize(ndims); // (rankx, ranky, rankz)
    ::MPI_Cart_coords(cart_comm, rank_, ndims, cart_rank_.data());

    int neighbors[6];
    ::MPI_Cart_shift(cart_comm, 0, 1, &neighbors[UP],   &neighbors[DOWN]);   // x direction
    ::MPI_Cart_shift(cart_comm, 1, 1, &neighbors[LEFT], &neighbors[RIGHT]);  // y direction
    ::MPI_Cart_shift(cart_comm, 2, 1, &neighbors[TOP],  &neighbors[BOTTOM]); // z direction

    bool is_comm_x = topology_.at(0) > 1;
    bool is_comm_y = topology_.at(1) > 1;
    bool is_comm_z = topology_.at(2) > 1;

    int left_tag = 0, right_tag = 1;

    using Shpae2D = shape_type<2>;
    x_send_halo_ = std::make_unique<HaloType>(q_, "x_send", Shpae2D({shape_[1], shape_[2]}), neighbors[UP], neighbors[DOWN], left_tag, right_tag, cart_comm, is_comm_x);
    x_recv_halo_ = std::make_unique<HaloType>(q_, "x_recv", Shpae2D({shape_[1], shape_[2]}), neighbors[UP], neighbors[DOWN], right_tag, left_tag, cart_comm, is_comm_x);

    y_send_halo_ = std::make_unique<HaloType>(q_, "y_send", Shpae2D({shape_[0], shape_[2]}), neighbors[LEFT], neighbors[RIGHT], left_tag, right_tag, cart_comm, is_comm_y);
    y_recv_halo_ = std::make_unique<HaloType>(q_, "y_recv", Shpae2D({shape_[0], shape_[2]}), neighbors[LEFT], neighbors[RIGHT], right_tag, left_tag, cart_comm, is_comm_y);

    z_send_halo_ = std::make_unique<HaloType>(q_, "z_send", Shpae2D({shape_[0], shape_[1]}), neighbors[TOP], neighbors[BOTTOM], left_tag, right_tag, cart_comm, is_comm_z);
    z_recv_halo_ = std::make_unique<HaloType>(q_, "z_recv", Shpae2D({shape_[0], shape_[1]}), neighbors[TOP], neighbors[BOTTOM], right_tag, left_tag, cart_comm, is_comm_z);
  }

public:
  template <class View>
  void exchangeHalos(View& u, std::vector<Timer*> &timers) {
    bool use_timer = timers.size() > 0;
    // Define submdspans for halo regions
    const std::pair inner_x(1, u.extent(0) - 1);
    const std::pair inner_y(1, u.extent(1) - 1);
    const std::pair inner_z(1, u.extent(2) - 1);

    // Exchange in x direction
    {
      auto ux_send_left  = std::submdspan(u, 1, inner_y, inner_z);
      auto ux_send_right = std::submdspan(u, u.extent(0) - 2, inner_y, inner_z);
      auto ux_recv_left  = std::submdspan(u, 0, inner_y, inner_z);
      auto ux_recv_right = std::submdspan(u, u.extent(0) - 1, inner_y, inner_z);

      if(use_timer) timers[HaloPack]->begin();
      pack_(x_send_halo_, ux_send_left, ux_send_right);
      if(use_timer) timers[HaloPack]->end();

      if(use_timer) timers[HaloComm]->begin();
      commP2P_(x_recv_halo_, x_send_halo_);
      if(use_timer) timers[HaloComm]->end();

      if(use_timer) timers[HaloUnpack]->begin();
      unpack_(ux_recv_left, ux_recv_right, x_recv_halo_);
      if(use_timer) timers[HaloUnpack]->end();
    }

    // Exchange in y direction
    {
      auto uy_send_left  = std::submdspan(u, inner_x, 1, inner_z);
      auto uy_send_right = std::submdspan(u, inner_x, u.extent(1) - 2, inner_z);
      auto uy_recv_left  = std::submdspan(u, inner_x, 0, inner_z);
      auto uy_recv_right = std::submdspan(u, inner_x, u.extent(1) - 1, inner_z);

      if(use_timer) timers[HaloPack]->begin();
      pack_(y_send_halo_, uy_send_left, uy_send_right);
      if(use_timer) timers[HaloPack]->end();

      if(use_timer) timers[HaloComm]->begin();
      commP2P_(y_recv_halo_, y_send_halo_);
      if(use_timer) timers[HaloComm]->end();

      if(use_timer) timers[HaloUnpack]->begin();
      unpack_(uy_recv_left, uy_recv_right, y_recv_halo_);
      if(use_timer) timers[HaloUnpack]->end();
    }

    // Exchange in z direction
    {
      auto uz_send_left  = std::submdspan(u, inner_x, inner_y, 1);
      auto uz_send_right = std::submdspan(u, inner_x, inner_y, u.extent(2) - 2);
      auto uz_recv_left  = std::submdspan(u, inner_x, inner_y, 0);
      auto uz_recv_right = std::submdspan(u, inner_x, inner_y, u.extent(2) - 1);

      if(use_timer) timers[HaloPack]->begin();
      pack_(z_send_halo_, uz_send_left, uz_send_right);
      if(use_timer) timers[HaloPack]->end();

      if(use_timer) timers[HaloComm]->begin();
      commP2P_(z_recv_halo_, z_send_halo_);
      if(use_timer) timers[HaloComm]->end();

      if(use_timer) timers[HaloUnpack]->begin();
      unpack_(uz_recv_left, uz_recv_right, z_recv_halo_);
      if(use_timer) timers[HaloUnpack]->end();
    }
  }

  void commP2P() {
    commP2P_(x_recv_halo_, x_send_halo_);
    commP2P_(y_recv_halo_, y_send_halo_);
    commP2P_(z_recv_halo_, z_send_halo_);
  }

private:
  template <class HaloType>
  void commP2P_(std::unique_ptr<HaloType>& recv,
                std::unique_ptr<HaloType>& send) {
    auto send_left = send->left(), send_right = send->right();
    auto recv_left = recv->left(), recv_right = recv->right();
    if(send->is_comm()) {
      MPI_Status status[4];
      MPI_Request request[4];
      MPI_Irecv(recv_left.data_handle(),  recv->size(), recv->type(), recv->left_rank(),  recv->left_tag(),  recv->communicator(), &request[0]);
      MPI_Irecv(recv_right.data_handle(), recv->size(), recv->type(), recv->right_rank(), recv->right_tag(), recv->communicator(), &request[1]);
      MPI_Isend(send_left.data_handle(),  send->size(), send->type(), send->left_rank(),  send->left_tag(),  send->communicator(), &request[2]);
      MPI_Isend(send_right.data_handle(), send->size(), send->type(), send->right_rank(), send->right_tag(), send->communicator(), &request[3]);

      MPI_Waitall( 4, request, status );
    } else {
      auto*& send_left  = send->left_ptr();
      auto*& send_right = send->right_ptr();
      auto*& recv_left  = recv->left_ptr();
      auto*& recv_right = recv->right_ptr();

      std::swap( send_left,  recv_right );
      std::swap( send_right, recv_left  );
    }
  }

  template <class HaloType, class View>
  void pack_(std::unique_ptr<HaloType>& send, const View& left, const View& right) {
    auto left_buffer = send->left();
    auto right_buffer = send->right();

    assert( left.extents() == right.extents() );
    assert( left.extents() == left_buffer.extents() );
    assert( left.extents() == right_buffer.extents() );

    // 2D loop
    auto n0 = left.extent(0), n1 = left.extent(1);
    sycl::range<2> global_range(n0, n1);
    sycl::range<2> local_range(4, 4);
    sycl::nd_range<2> nd_range(global_range, local_range);

    q_.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
        nd_range, 
        Copyfunctor(left, right, left_buffer, right_buffer)
      );
    }).wait();
  }

  template <class HaloType, class View>
  void unpack_(View& left, View& right, std::unique_ptr<HaloType>& recv) {
    const auto left_buffer = recv->left();
    const auto right_buffer = recv->right();

    assert( left.extents() == right.extents() );
    assert( left.extents() == left_buffer.extents() );
    assert( left.extents() == right_buffer.extents() );

    // 2D loop
    auto n0 = left.extent(0), n1 = left.extent(1);
    sycl::range<2> global_range(n0, n1);
    sycl::range<2> local_range(4, 4);
    sycl::nd_range<2> nd_range(global_range, local_range);

    q_.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
        nd_range, 
        Copyfunctor(left_buffer, right_buffer, left, right)
      );
    }).wait();
  }
};

/* Helpers to update boundaries */
template <class HaloType, class View>
void async_pack(sycl::queue& q, std::unique_ptr<HaloType>& send, const View& left, const View& right) {
  auto left_buffer  = send->left();
  auto right_buffer = send->right();

  assert( left.extents() == right.extents() );
  assert( left.extents() == left_buffer.extents() );
  assert( left.extents() == right_buffer.extents() );

  auto n0 = left.extent(0), n1 = left.extent(1);
  sycl::range<2> global_range(n0, n1);
  sycl::range<2> local_range(4, 4);
  sycl::nd_range<2> nd_range(global_range, local_range);

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
      nd_range, 
      Copyfunctor(left, right, left_buffer, right_buffer)
    );
  });
}

template <class HaloType, class View>
void async_unpack(sycl::queue& q, View& left, View& right, std::unique_ptr<HaloType>& recv) {
  const auto left_buffer  = recv->left();
  const auto right_buffer = recv->right();

  assert( left.extents() == right.extents() );
  assert( left.extents() == left_buffer.extents() );
  assert( left.extents() == right_buffer.extents() );

  auto n0 = left.extent(0), n1 = left.extent(1);
  sycl::range<2> global_range(n0, n1);
  sycl::range<2> local_range(4, 4);
  sycl::nd_range<2> nd_range(global_range, local_range);

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
      nd_range, 
      Copyfunctor(left_buffer, right_buffer, left, right)
    );
  });
}

template <class HaloType, class View>
void async_boundaryUpdate(sycl::queue& q, const Config& conf, View& left, View& right, std::unique_ptr<HaloType>& recv) {
  const auto left_buffer  = recv->left();
  const auto right_buffer = recv->right();

  assert( left.extents() == right.extents() );
  assert( left.extents() == left_buffer.extents() );
  assert( left.extents() == right_buffer.extents() );

  auto n0 = left.extent(0), n1 = left.extent(1);
  sycl::range<2> global_range(n0, n1);
  sycl::range<2> local_range(4, 4);
  sycl::nd_range<2> nd_range(global_range, local_range);

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
      nd_range, 
      Heat3DBoundaryfunctor(conf, left_buffer, right_buffer, left, right)
    );
  });
}

template <class View>
void async_pack_all(sycl::queue& q, Comm& comm, View& u) {
  // Define submdspans for halo regions
  const std::pair inner_x(1, u.extent(0) - 1);
  const std::pair inner_y(1, u.extent(1) - 1);
  const std::pair inner_z(1, u.extent(2) - 1);

  auto ux_send_left  = std::submdspan(u, 1, inner_y, inner_z);
  auto ux_send_right = std::submdspan(u, u.extent(0) - 2, inner_y, inner_z);

  auto uy_send_left  = std::submdspan(u, inner_x, 1, inner_z);
  auto uy_send_right = std::submdspan(u, inner_x, u.extent(1) - 2, inner_z);

  auto uz_send_left  = std::submdspan(u, inner_x, inner_y, 1);
  auto uz_send_right = std::submdspan(u, inner_x, inner_y, u.extent(2) - 2);

  async_pack(q, comm.send_buffer(0), ux_send_left, ux_send_right);
  async_pack(q, comm.send_buffer(1), uy_send_left, uy_send_right);
  async_pack(q, comm.send_buffer(2), uz_send_left, uz_send_right);
}

template <class View>
void async_unpack_all(sycl::queue& q, Comm& comm, View& u) {
  // Define submdspans for halo regions
  const std::pair inner_x(1, u.extent(0) - 1);
  const std::pair inner_y(1, u.extent(1) - 1);
  const std::pair inner_z(1, u.extent(2) - 1);

  auto ux_recv_left  = std::submdspan(u, 0, inner_y, inner_z);
  auto ux_recv_right = std::submdspan(u, u.extent(0) - 1, inner_y, inner_z);

  auto uy_recv_left  = std::submdspan(u, inner_x, 0, inner_z);
  auto uy_recv_right = std::submdspan(u, inner_x, u.extent(1) - 1, inner_z);

  auto uz_recv_left  = std::submdspan(u, inner_x, inner_y, 0);
  auto uz_recv_right = std::submdspan(u, inner_x, inner_y, u.extent(2) - 1);

  async_unpack(q, ux_recv_left, ux_recv_right, comm.recv_buffer(0));
  async_unpack(q, uy_recv_left, uy_recv_right, comm.recv_buffer(1));
  async_unpack(q, uz_recv_left, uz_recv_right, comm.recv_buffer(2));
}

template <class View>
void async_boundaryUpdate_all(sycl::queue& q, const Config& conf, Comm& comm, View& u) {
  // Define submdspans for halo regions
  const std::pair inner_x(1, u.extent(0) - 1);
  const std::pair inner_y(1, u.extent(1) - 1);
  const std::pair inner_z(1, u.extent(2) - 1);

  auto ux_recv_left  = std::submdspan(u, 1, inner_y, inner_z);
  auto ux_recv_right = std::submdspan(u, u.extent(0) - 2, inner_y, inner_z);

  auto uy_recv_left  = std::submdspan(u, inner_x, 1, inner_z);
  auto uy_recv_right = std::submdspan(u, inner_x, u.extent(1) - 2, inner_z);

  auto uz_recv_left  = std::submdspan(u, inner_x, inner_y, 1);
  auto uz_recv_right = std::submdspan(u, inner_x, inner_y, u.extent(2) - 2);

  async_boundaryUpdate(q, conf, ux_recv_left, ux_recv_right, comm.recv_buffer(0));
  q.wait();
  async_boundaryUpdate(q, conf, uy_recv_left, uy_recv_right, comm.recv_buffer(1));
  q.wait();
  async_boundaryUpdate(q, conf, uz_recv_left, uz_recv_right, comm.recv_buffer(2));
  q.wait();
}

#endif