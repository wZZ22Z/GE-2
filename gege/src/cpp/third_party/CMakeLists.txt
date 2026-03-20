function(initialize_submodule DIRECTORY)
    # In some distributions (e.g. source tarballs / vendored deps) we may have a
    # populated dependency directory without its `.git` metadata. Only attempt
    # to init/update the submodule when the dependency sources are missing.
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${DIRECTORY}/CMakeLists.txt)
        find_package(Git QUIET REQUIRED)
        message(STATUS "${CMAKE_CURRENT_SOURCE_DIR}/${DIRECTORY}/.git does not exist. Initializing ${DIRECTORY} submodule ...")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init ${DIRECTORY}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_EXIT_CODE)
        if(NOT GIT_EXIT_CODE EQUAL "0")
            message(FATAL_ERROR "${GIT_EXECUTABLE} submodule update --init dependencies/${DIRECTORY} failed with exit code ${GIT_EXIT_CODE}, please checkout submodules")
        endif()
    endif()
endfunction(initialize_submodule)

initialize_submodule(pybind11)
initialize_submodule(spdlog)
initialize_submodule(parallel-hashmap)

add_subdirectory(pybind11 EXCLUDE_FROM_ALL)
add_subdirectory(spdlog EXCLUDE_FROM_ALL)
