cmake_minimum_required(VERSION 3.22)
cmake_policy(SET CMP0048 NEW)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(ge2 VERSION 1.0 LANGUAGES CXX)

include(FindPackageHandleStandardArgs)

if(DEFINED PYTHON_EXECUTABLE)
    set(GEGE_PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}")
else()
    set(GEGE_PYTHON_EXECUTABLE python3)
endif()

set(CMAKE_CXX_VISIBILITY_PRESET default)

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
        message(FATAL_ERROR "GCC version must be at least 7.0!")
    endif()
    set(GCC TRUE)
else ()
    message(FATAL_ERROR "Unknown compiler")
endif ()

if(${USE_CUDA})
    add_definitions(-DGEGE_CUDA=${USE_CUDA})
    # PyTorch/ATen headers require C++17 (including when compiling CUDA .cu
    # sources via NVCC).
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
    enable_language(CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

# Find torch location
execute_process(
        COMMAND ${GEGE_PYTHON_EXECUTABLE} -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
        OUTPUT_VARIABLE TorchPath
)
list(APPEND CMAKE_PREFIX_PATH ${TorchPath})

execute_process(
        COMMAND ${GEGE_PYTHON_EXECUTABLE} -c "import torch; print(torch.__version__, end='')"
        OUTPUT_VARIABLE TorchVersion
)

message(STATUS "Torch Version: ${TorchVersion}")

set(GEGE_CPP_SOURCE ${CMAKE_CURRENT_LIST_DIR}/src/cpp)
set(project_INCLUDE_DIR ${GEGE_CPP_SOURCE}/include)
set(project_SOURCE_DIR ${GEGE_CPP_SOURCE}/src)
set(project_CUDA_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/src/cuda/include)
set(project_CUDA_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/src/cuda/src)
set(project_CUDA_THIRD_PARTY_DIR ${CMAKE_CURRENT_LIST_DIR}/src/cuda/third_party)
set(project_BINDINGS_DIR ${GEGE_CPP_SOURCE}/python_bindings)
set(project_THIRD_PARTY_DIR ${GEGE_CPP_SOURCE}/third_party)

set(project_WORKING_DIR ${CMAKE_CURRENT_BINARY_DIR})
add_definitions(-DGEGE_BASE_DIRECTORY="${CMAKE_CURRENT_LIST_DIR}")

if (EXISTS ${project_INCLUDE_DIR})
    file(GLOB_RECURSE project_HEADERS ${project_HEADERS} ${project_INCLUDE_DIR}/*.h)
endif ()
if (EXISTS ${project_SOURCE_DIR})
    file(GLOB_RECURSE project_SOURCES ${project_SOURCES} ${project_SOURCE_DIR}/*.cpp)
    list(REMOVE_ITEM project_SOURCES ${project_SOURCE_DIR}/gege_unique_replay_bench.cpp)
    list(REMOVE_ITEM project_SOURCES ${project_SOURCE_DIR}/gege_cover_schedule_analyzer.cpp)
    list(REMOVE_ITEM project_SOURCES ${project_SOURCE_DIR}/gege_partition_access_analyzer.cpp)
    list(REMOVE_ITEM project_SOURCES ${project_SOURCE_DIR}/gege_hot_entity_access_analyzer.cpp)
    list(REMOVE_ITEM project_SOURCES ${project_SOURCE_DIR}/gege_delta_log_update_analyzer.cpp)
    list(REMOVE_ITEM project_SOURCES ${project_SOURCE_DIR}/gege_partition_handoff_analyzer.cpp)
endif ()

if(${USE_CUDA})
    if (EXISTS ${project_CUDA_INCLUDE_DIR})
        file(GLOB_RECURSE project_CUDA_HEADERS ${project_CUDA_INCLUDE_DIR} ${project_CUDA_INCLUDE_DIR}/*.cuh)
    endif ()
    if (EXISTS ${project_CUDA_SOURCE_DIR})
        file(GLOB_RECURSE project_CUDA_SOURCES ${project_CUDA_SOURCE_DIR} ${project_CUDA_SOURCE_DIR}/*.cu)
    endif ()

    if (EXISTS ${project_CUDA_THIRD_PARTY_DIR})
        file(GLOB_RECURSE project_CUDA_THIRD_PARTY_HEADERS ${project_CUDA_THIRD_PARTY_DIR} ${project_CUDA_THIRD_PARTY_DIR}/*.cuh ${project_CUDA_THIRD_PARTY_DIR}/*.h)
    endif ()
    if (EXISTS ${project_CUDA_THIRD_PARTY_DIR})
        file(GLOB_RECURSE project_CUDA_THIRD_PARTY_SOURCES ${project_CUDA_THIRD_PARTY_DIR} ${project_CUDA_THIRD_PARTY_DIR}/*.cu ${project_CUDA_THIRD_PARTY_DIR}/*.cpp)
    endif ()
endif ()

message(STATUS "project_CUDA_THIRD_PARTY_HEADERS ${project_CUDA_THIRD_PARTY_HEADERS}")
message(STATUS "project_CUDA_THIRD_PARTY_SOURCES ${project_CUDA_THIRD_PARTY_SOURCES}")

find_package(Python3 COMPONENTS Development Interpreter REQUIRED)
# pybind11 (vendored) uses CMake's "new" Python mode whenever Python3_FOUND is
# set, which expects `python_add_library()` from `find_package(Python ...)`.
set(Python_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH "" FORCE)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(Torch REQUIRED)

execute_process(
        COMMAND ${GEGE_PYTHON_EXECUTABLE} -c "import torch; print(int(bool(torch._C._GLIBCXX_USE_CXX11_ABI)), end='')"
        OUTPUT_VARIABLE TORCH_CXX11_ABI
)
if(NOT TORCH_CXX11_ABI STREQUAL "0" AND NOT TORCH_CXX11_ABI STREQUAL "1")
    set(TORCH_CXX11_ABI 1)
endif()
message(STATUS "TORCH _GLIBCXX_USE_CXX11_ABI: ${TORCH_CXX11_ABI}")
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})

execute_process(
        COMMAND ${GEGE_PYTHON_EXECUTABLE} -c "import torch; print(getattr(torch._C, '_PYBIND11_COMPILER_TYPE', ''), end='')"
        OUTPUT_VARIABLE _PYBIND11_COMPILER_TYPE
)
execute_process(
        COMMAND ${GEGE_PYTHON_EXECUTABLE} -c "import torch; print(getattr(torch._C, '_PYBIND11_STDLIB', ''), end='')"
        OUTPUT_VARIABLE _PYBIND11_STDLIB
)
execute_process(
        COMMAND ${GEGE_PYTHON_EXECUTABLE} -c "import torch; print(getattr(torch._C, '_PYBIND11_BUILD_ABI', ''), end='')"
        OUTPUT_VARIABLE _PYBIND11_BUILD_ABI
)

message(STATUS "PYBIND11_COMPILER_TYPE:" ${_PYBIND11_COMPILER_TYPE})
message(STATUS "PYBIND11_STDLIB:" ${_PYBIND11_STDLIB})
message(STATUS "PYBIND11_BUILD_ABI:" ${_PYBIND11_BUILD_ABI})

if(NOT _PYBIND11_COMPILER_TYPE STREQUAL "" OR NOT _PYBIND11_STDLIB STREQUAL "" OR NOT _PYBIND11_BUILD_ABI STREQUAL "")
    add_compile_definitions(
            PYBIND11_COMPILER_TYPE="${_PYBIND11_COMPILER_TYPE}"
            PYBIND11_STDLIB="${_PYBIND11_STDLIB}"
            PYBIND11_BUILD_ABI="${_PYBIND11_BUILD_ABI}")
else()
    message(STATUS "Torch does not export PYBIND11_* ABI tags; using pybind11 defaults.")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

message(STATUS "Python3_INCLUDE_DIRS ${Python3_INCLUDE_DIRS}")
add_subdirectory(${project_THIRD_PARTY_DIR})
set_property(TARGET spdlog PROPERTY POSITION_INDEPENDENT_CODE ON)

include_directories(${Python3_INCLUDE_DIRS})
include_directories(${project_INCLUDE_DIR})
include_directories(${project_CUDA_INCLUDE_DIR})
include_directories(${project_CUDA_THIRD_PARTY_DIR})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${project_THIRD_PARTY_DIR}/parallel-hashmap/)
include_directories(${project_BINDINGS})

add_library(${PROJECT_NAME}
            SHARED
            ${project_SOURCES}
            ${project_HEADERS}
            ${project_CUDA_HEADERS}
            ${project_CUDA_SOURCES}
            ${project_CUDA_THIRD_PARTY_HEADERS}
            ${project_CUDA_THIRD_PARTY_SOURCES})

target_link_libraries(${PROJECT_NAME} ${Python3_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
target_link_libraries(${PROJECT_NAME} spdlog)
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${project_HEADERS}")
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(${USE_CUDA})
    target_compile_options(${PROJECT_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
endif()

if(${USE_OMP})
    add_definitions(-DGEGE_OMP=${USE_OMP})
    if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
        set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
        set(OpenMP_CXX_FLAGS "-fopenmp")
    endif()
    find_package(OpenMP REQUIRED)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()

if (EXISTS ${project_INCLUDE_DIR})
    target_include_directories(${PROJECT_NAME} PUBLIC ${project_INCLUDE_DIR})
endif ()
if (EXISTS ${project_SOURCE_DIR})
    target_include_directories(${PROJECT_NAME} PRIVATE ${project_SOURCE_DIR})
endif ()

IF(CMAKE_BUILD_TYPE MATCHES Debug AND GEGE_USE_TSAN)
    message("Using address sanitizer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
    set(CMAKE_MODULE_LINKER_FLAGS "{$CMAKE_MODULE_LINKER_FLAGS} -fsanitize=thread")
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug AND GEGE_USE_TSAN)

IF(CMAKE_BUILD_TYPE MATCHES Debug AND GEGE_USE_ASAN)
    message("Using thread sanitizer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=leak")
    set(CMAKE_MODULE_LINKER_FLAGS "{$CMAKE_MODULE_LINKER_FLAGS} -fsanitize=address -fsanitize=leak")
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug AND GEGE_USE_ASAN)


add_executable(gege_train ${project_SOURCE_DIR}/gege.cpp)
add_executable(gege_eval ${project_SOURCE_DIR}/gege.cpp)
add_executable(gege_unique_replay_bench ${project_SOURCE_DIR}/gege_unique_replay_bench.cpp)
add_executable(gege_cover_schedule_analyzer ${project_SOURCE_DIR}/gege_cover_schedule_analyzer.cpp)
add_executable(gege_partition_access_analyzer ${project_SOURCE_DIR}/gege_partition_access_analyzer.cpp)
add_executable(gege_hot_entity_access_analyzer ${project_SOURCE_DIR}/gege_hot_entity_access_analyzer.cpp)
add_executable(gege_delta_log_update_analyzer ${project_SOURCE_DIR}/gege_delta_log_update_analyzer.cpp)
add_executable(gege_partition_handoff_analyzer ${project_SOURCE_DIR}/gege_partition_handoff_analyzer.cpp)
target_link_libraries(gege_train ${PROJECT_NAME})
target_link_libraries(gege_eval ${PROJECT_NAME})
target_link_libraries(gege_unique_replay_bench ${PROJECT_NAME})
target_link_libraries(gege_cover_schedule_analyzer ${PROJECT_NAME})
target_link_libraries(gege_partition_access_analyzer ${PROJECT_NAME})
target_link_libraries(gege_hot_entity_access_analyzer ${PROJECT_NAME})
target_link_libraries(gege_delta_log_update_analyzer ${PROJECT_NAME})
target_link_libraries(gege_partition_handoff_analyzer ${PROJECT_NAME})

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")

file(GLOB_RECURSE CONFIG_BINDINGS ${project_BINDINGS} ${project_BINDINGS_DIR}/configuration/*.cpp)
pybind11_add_module(_config ${CONFIG_BINDINGS})
target_link_libraries(_config PRIVATE ${PROJECT_NAME} ${TORCH_PYTHON_LIBRARY})

file(GLOB_RECURSE DATA_BINDINGS ${project_BINDINGS} ${project_BINDINGS_DIR}/data/*.cpp)
pybind11_add_module(_data ${DATA_BINDINGS})
target_link_libraries(_data PRIVATE ${PROJECT_NAME} ${TORCH_PYTHON_LIBRARY})

file(GLOB_RECURSE NN_BINDINGS ${project_BINDINGS} ${project_BINDINGS_DIR}/nn/*.cpp)
pybind11_add_module(_nn ${NN_BINDINGS})
target_link_libraries(_nn PRIVATE ${PROJECT_NAME} ${TORCH_PYTHON_LIBRARY})

file(GLOB_RECURSE MANAGER_BINDINGS ${project_BINDINGS} ${project_BINDINGS_DIR}/manager/*.cpp)
pybind11_add_module(_manager ${MANAGER_BINDINGS})
target_link_libraries(_manager PRIVATE ${PROJECT_NAME} ${TORCH_PYTHON_LIBRARY})

file(GLOB_RECURSE ENGINE_BINDINGS ${project_BINDINGS} ${project_BINDINGS_DIR}/engine/*.cpp)
pybind11_add_module(_engine ${ENGINE_BINDINGS})
target_link_libraries(_engine PRIVATE ${PROJECT_NAME} ${TORCH_PYTHON_LIBRARY})

file(GLOB_RECURSE REPORT_BINDINGS ${project_BINDINGS} ${project_BINDINGS_DIR}/reporting/*.cpp)
pybind11_add_module(_report ${REPORT_BINDINGS})
target_link_libraries(_report PRIVATE ${PROJECT_NAME} ${TORCH_PYTHON_LIBRARY})

file(GLOB_RECURSE STORAGE_BINDINGS ${project_BINDINGS} ${project_BINDINGS_DIR}/storage/*.cpp)
pybind11_add_module(_storage ${STORAGE_BINDINGS})
target_link_libraries(_storage PRIVATE ${PROJECT_NAME} ${TORCH_PYTHON_LIBRARY})

add_custom_target(bindings)
add_dependencies(bindings _config _data _manager _nn _engine _report _storage)
