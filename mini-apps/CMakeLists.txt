if(APPLICATION STREQUAL "heat3d")
    add_subdirectory(heat3d)
elseif(APPLICATION STREQUAL "heat3d_mpi")
    add_subdirectory(heat3d_mpi)
else()
    # Default: all mini-apps
    add_subdirectory(heat3d)
    add_subdirectory(heat3d_mpi)
endif()
