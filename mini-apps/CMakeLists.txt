if(APPLICATION STREQUAL "heat3d")
    add_subdirectory(heat3d)
elseif(APPLICATION STREQUAL "heat3d-mpi")
    add_subdirectory(heat3d-mpi)
else()
    # Default: all mini-apps
    add_subdirectory(heat3d)
    add_subdirectory(heat3d-mpi)
endif()
