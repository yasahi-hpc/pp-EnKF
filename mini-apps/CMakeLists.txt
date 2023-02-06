if(APPLICATION STREQUAL "heat3d")
    add_subdirectory(heat3d)
else()
    # Default: all mini-apps
    add_subdirectory(heat3d)
endif()
