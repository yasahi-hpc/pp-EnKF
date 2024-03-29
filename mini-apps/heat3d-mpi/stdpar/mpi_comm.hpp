#ifndef __MPI_COMM_HPP__
#define __MPI_COMM_HPP__

#include <cassert>
#include <vector>
#include <complex>
#include <mpi.h>
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
  using Vector = std::vector<RealType>;

private:
  Vector left_, right_;
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
  Halo(const std::string name, Shpae2D shape, int left_rank, int right_rank,
       int left_tag, int right_tag, MPI_Comm communicator, bool is_comm)
    : name_(name), extents_(shape), left_rank_(left_rank), right_rank_(right_rank),
    left_tag_(left_tag), right_tag_(right_tag), communicator_(communicator), is_comm_(is_comm) {

    size_ = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<std::size_t>());
    left_.resize(size_, 0); right_.resize(size_, 0);

    mpi_data_type_ = get_mpi_data_type<RealType>();
  }

  Halo(Halo&& rhs) noexcept
    : name_(rhs.name_), extents_(rhs.extents_), size_(rhs.size_), left_rank_(rhs.left_rank_), right_rank_(rhs.right_rank_),
    left_(rhs.left_), right_(rhs.right_), left_tag_(rhs.left_tag_), right_tag_(rhs.right_tag_), 
    communicator_(rhs.communicator_), mpi_data_type_(rhs.mpi_data_type_), is_comm_(rhs.is_comm_) {
  }

  ~Halo() {}

  // Getters
  RealView2D left() {
    return RealView2D(left_.data(), extents_);
  }

  RealView2D right() {
    return RealView2D(right_.data(), extents_);
  }

  Vector left_vector() const { return left_; }
  Vector right_vector() const { return right_; }
  Vector& left_vector() { return left_; }
  Vector& right_vector() { return right_; }

  std::size_t size() const { return size_; }
  int left_rank() const { return left_rank_; }
  int right_rank() const { return right_rank_; }
  int left_tag() const { return left_tag_; }
  int right_tag() const { return right_tag_; }
  MPI_Comm communicator() const { return communicator_; }
  MPI_Datatype type() const { return mpi_data_type_; }
  bool is_comm() const {return is_comm_; }

private:
  DISALLOW_COPY_AND_ASSIGN(Halo);
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
  std::vector< Halo<double> > send_halos_;
  std::vector< Halo<double> > recv_halos_;

  int halo_width_;

public:
  Comm() = delete;
  Comm(int& argc,
       char** argv,
       const std::vector<std::size_t>& shape,
       const std::vector<int>& topology)
    : shape_(shape), topology_(topology), halo_width_(1) {
    int required = MPI_THREAD_MULTIPLE;
    int provided;
    ::MPI_Init_thread(&argc, &argv, required, &provided);
    ::MPI_Comm_size(MPI_COMM_WORLD, &size_);
    ::MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

    setDevice( rank_ );
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

  Halo<double>& send_buffer(size_t i) {
    assert( i < send_halos_.size() );
    return send_halos_.at(i);
  }

  Halo<double>& recv_buffer(size_t i) {
    assert( i < recv_halos_.size() );
    return recv_halos_.at(i);
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
    send_halos_.push_back( Halo<double>("x_send", {shape_[1], shape_[2]}, neighbors[UP], neighbors[DOWN], left_tag, right_tag, cart_comm, is_comm_x) );
    recv_halos_.push_back( Halo<double>("x_recv", {shape_[1], shape_[2]}, neighbors[UP], neighbors[DOWN], right_tag, left_tag, cart_comm, is_comm_x) );

    send_halos_.push_back( Halo<double>("y_send", {shape_[0], shape_[2]}, neighbors[LEFT], neighbors[RIGHT], left_tag, right_tag, cart_comm, is_comm_y) );
    recv_halos_.push_back( Halo<double>("y_recv", {shape_[0], shape_[2]}, neighbors[LEFT], neighbors[RIGHT], right_tag, left_tag, cart_comm, is_comm_y) );

    send_halos_.push_back( Halo<double>("z_send", {shape_[0], shape_[1]}, neighbors[TOP], neighbors[BOTTOM], left_tag, right_tag, cart_comm, is_comm_z) );
    recv_halos_.push_back( Halo<double>("z_recv", {shape_[0], shape_[1]}, neighbors[TOP], neighbors[BOTTOM], right_tag, left_tag, cart_comm, is_comm_z) );
  }

public:
  void commP2P() {
    for(std::size_t i=0; i<send_halos_.size(); i++) {
      commP2P_(recv_halos_.at(i), send_halos_.at(i));
    }
  }

  template <class View>
  void exchangeHalos(View& u, std::vector<Timer*> &timers) {
    bool use_timer = timers.size() > 0;
    // Define submdspans for halo regions
    const std::pair inner_x(1, u.extent(0) - 1);
    const std::pair inner_y(1, u.extent(1) - 1);
    const std::pair inner_z(1, u.extent(2) - 1);

    // Exchange in x direction
    {
      int i = 0;
      auto ux_send_left  = std::submdspan(u, 1, inner_y, inner_z);
      auto ux_send_right = std::submdspan(u, u.extent(0) - 2, inner_y, inner_z);
      auto ux_recv_left  = std::submdspan(u, 0, inner_y, inner_z);
      auto ux_recv_right = std::submdspan(u, u.extent(0) - 1, inner_y, inner_z);

      if(use_timer) timers[HaloPack]->begin();
      pack_(send_buffer(i), ux_send_left, ux_send_right);
      if(use_timer) timers[HaloPack]->end();

      if(use_timer) timers[HaloComm]->begin();
      commP2P_(recv_buffer(i), send_buffer(i));
      if(use_timer) timers[HaloComm]->end();

      if(use_timer) timers[HaloUnpack]->begin();
      unpack_(ux_recv_left, ux_recv_right, recv_buffer(i));
      if(use_timer) timers[HaloUnpack]->end();
    }

    // Exchange in y direction
    {
      int i = 1;
      auto uy_send_left  = std::submdspan(u, inner_x, 1, inner_z);
      auto uy_send_right = std::submdspan(u, inner_x, u.extent(1) - 2, inner_z);
      auto uy_recv_left  = std::submdspan(u, inner_x, 0, inner_z);
      auto uy_recv_right = std::submdspan(u, inner_x, u.extent(1) - 1, inner_z);

      if(use_timer) timers[HaloPack]->begin();
      pack_(send_buffer(i), uy_send_left, uy_send_right);
      if(use_timer) timers[HaloPack]->end();

      if(use_timer) timers[HaloComm]->begin();
      commP2P_(recv_buffer(i), send_buffer(i));
      if(use_timer) timers[HaloComm]->end();

      if(use_timer) timers[HaloUnpack]->begin();
      unpack_(uy_recv_left, uy_recv_right, recv_buffer(i));
      if(use_timer) timers[HaloUnpack]->end();
    }

    // Exchange in z direction
    {
      int i = 2;
      auto uz_send_left  = std::submdspan(u, inner_x, inner_y, 1);
      auto uz_send_right = std::submdspan(u, inner_x, inner_y, u.extent(2) - 2);
      auto uz_recv_left  = std::submdspan(u, inner_x, inner_y, 0);
      auto uz_recv_right = std::submdspan(u, inner_x, inner_y, u.extent(2) - 1);

      if(use_timer) timers[HaloPack]->begin();
      pack_(send_buffer(i), uz_send_left, uz_send_right);
      if(use_timer) timers[HaloPack]->end();

      if(use_timer) timers[HaloComm]->begin();
      commP2P_(recv_buffer(i), send_buffer(i));
      if(use_timer) timers[HaloComm]->end();

      if(use_timer) timers[HaloUnpack]->begin();
      unpack_(uz_recv_left, uz_recv_right, recv_buffer(i));
      if(use_timer) timers[HaloUnpack]->end();
    }
  }

private:
  template <class HaloType>
  void commP2P_(HaloType& recv, HaloType& send) {
    auto send_left = send.left(), send_right = send.right();
    auto recv_left = recv.left(), recv_right = recv.right();
    if(send.is_comm()) {
      MPI_Status status[4];
      MPI_Request request[4];
      MPI_Irecv(recv_left.data_handle(),  recv.size(), recv.type(), recv.left_rank(),  recv.left_tag(),  recv.communicator(), &request[0]);
      MPI_Irecv(recv_right.data_handle(), recv.size(), recv.type(), recv.right_rank(), recv.right_tag(), recv.communicator(), &request[1]);
      MPI_Isend(send_left.data_handle(),  send.size(), send.type(), send.left_rank(),  send.left_tag(),  send.communicator(), &request[2]);
      MPI_Isend(send_right.data_handle(), send.size(), send.type(), send.right_rank(), send.right_tag(), send.communicator(), &request[3]);

      MPI_Waitall( 4, request, status );
    } else {
      std::vector<double>& send_left_vector  = send.left_vector();
      std::vector<double>& send_right_vector = send.right_vector();
      std::vector<double>& recv_left_vector  = recv.left_vector();
      std::vector<double>& recv_right_vector = recv.right_vector();

      std::swap( send_left_vector, recv_right_vector );
      std::swap( send_right_vector, recv_left_vector );
    }
  }

  template <class HaloType, class View>
  void pack_(HaloType& send, const View& left, const View& right) {
    auto left_buffer = send.left();
    auto right_buffer = send.right();
    const std::size_t n = left.size();

    assert( left.extents() == right.extents() );
    assert( left.extents() == left_buffer.extents() );
    assert( left.extents() == right_buffer.extents() );

    std::for_each_n(std::execution::par_unseq,
                    std::views::iota(0).begin(), n,
                    copy_functor(left, right, left_buffer, right_buffer));
  }

  template <class HaloType, class View>
  void unpack_(View& left, View& right, HaloType& recv) {
    const auto left_buffer = recv.left();
    const auto right_buffer = recv.right();
    const std::size_t n = left.size();

    assert( left.extents() == right.extents() );
    assert( left.extents() == left_buffer.extents() );
    assert( left.extents() == right_buffer.extents() );

    std::for_each_n(std::execution::par_unseq,
                    std::views::iota(0).begin(), n,
                    copy_functor(left_buffer, right_buffer, left, right));
  }
  
};

#endif
