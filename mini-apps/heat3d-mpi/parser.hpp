#ifndef __PARSER_HPP__
#define __PARSER_HPP__

#include <cstdio>
#include <string>
#include <string.h>
#include <vector>

struct Parser {
  std::vector<size_t> shape_;
  std::vector<int> topology_;

  int nbiter_ = 1000;
  int freq_diag_ = 10;
  int num_threads_ = 1;
  int teams_ = 1;
  int device_ = 0;
  int ngpu_ = 1;
  bool is_async_ = false;
  bool use_time_stamps_ = false;

  Parser() = delete;
  Parser(int argc, char** argv) {
    shape_.resize(3);
    topology_.resize(3);
    size_t nx = 512, ny = 512, nz = 512;
    int px = 2, py = 2, pz = 2;

    for(int i = 0; i < argc; i++) {
      if((strcmp(argv[i], "-nx") == 0) || (strcmp(argv[i], "--nx") == 0)) {
        nx = static_cast<size_t>( atoi(argv[++i]) );
        continue;
      }

      if((strcmp(argv[i], "-ny") == 0) || (strcmp(argv[i], "--ny") == 0)) {
        ny = static_cast<size_t>( atoi(argv[++i]) );
        continue;
      }

      if((strcmp(argv[i], "-nz") == 0) || (strcmp(argv[i], "--nz") == 0)) {
        nz = static_cast<size_t>( atoi(argv[++i]) );
        continue;
      }

      if((strcmp(argv[i], "-px") == 0) || (strcmp(argv[i], "--px") == 0)) {
        px = atoi(argv[++i]);
        continue;
      }

      if((strcmp(argv[i], "-py") == 0) || (strcmp(argv[i], "--py") == 0)) {
        py = atoi(argv[++i]);
        continue;
      }

      if((strcmp(argv[i], "-pz") == 0) || (strcmp(argv[i], "--pz") == 0)) {
        pz = atoi(argv[++i]);
        continue;
      }

      if((strcmp(argv[i], "-nbiter") == 0) || (strcmp(argv[i], "--nbiter") == 0)) {
        nbiter_ = atoi(argv[++i]);
        continue;
      }

      if((strcmp(argv[i], "-freq_diag") == 0) || (strcmp(argv[i], "--freq_diag") == 0)) {
        freq_diag_ = atoi(argv[++i]);
        continue;
      }

      if((strcmp(argv[i], "-is_async") == 0) || (strcmp(argv[i], "--is_async") == 0)) {
        int is_async = atoi(argv[++i]);
        is_async_ = is_async >= 1;
        continue;
      }

      if((strcmp(argv[i], "-use_time_stamps") == 0) || (strcmp(argv[i], "--use_time_stamps") == 0)) {
        int use_time_stamps = atoi(argv[++i]);
        use_time_stamps_ = use_time_stamps >= 1;
        continue;
      }

      if((strcmp(argv[i], "-t") == 0) || (strcmp(argv[i], "--num_threads") == 0)) {
        num_threads_ = atoi(argv[++i]);
        continue;
      }

      if((strcmp(argv[i], "--teams") == 0)) {
        teams_ = atoi(argv[++i]);
        continue;
      }

      if((strcmp(argv[i], "-d") == 0) || (strcmp(argv[i], "--device") == 0)) {
        device_ = atoi(argv[++i]);
        continue;
      }

      if((strcmp(argv[i], "-ng") == 0) || (strcmp(argv[i], "--num_gpus") == 0)) {
        ngpu_ = atoi(argv[++i]);
        continue;
      }

      if((strcmp(argv[i], "-dm") == 0) || (strcmp(argv[i], "--device_map") == 0)) {
        char *str;
        int local_rank;

        if((str = getenv("SLURM_LOCALID")) != NULL) {
          local_rank = atoi(str);
          device_ = local_rank % ngpu_;
        }

        if((str = getenv("MV2_COMM_WORLD_LOCAL_RANK")) != NULL) {
          local_rank = atoi(str);
          device_ = local_rank % ngpu_;
        }

        if((str = getenv("OMPI_COMM_WORLD_LOCAL_RANK")) != NULL) {
          local_rank = atoi(str);
          device_ = local_rank % ngpu_;
        }

        if((str = getenv("MPT_LRANK")) != NULL) {
          local_rank = atoi(str);
          device_ = local_rank % ngpu_;
        }

        continue;
      }
    }
    shape_    = std::vector<size_t>({nx, ny, nz});
    topology_ = std::vector<int>({px, py, pz});
  }
  ~Parser() {}
};

#endif
