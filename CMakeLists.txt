cmake_minimum_required(VERSION 3.20)

project(gege VERSION 0.1 LANGUAGES CXX)

# find python3
find_package(Python3 3.6 COMPONENTS Interpreter Development)
if (NOT Python3_FOUND)
    message(FATAL_ERROR "Python not found. GE^2 requires Python >= 3.6.")
endif()

# set variables for gege
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE} CACHE FILEPATH "" FORCE)
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/package/python CACHE PATH "" FORCE)
set(PACKAGE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/package CACHE PATH "" FORCE)
set(USE_CUDA TRUE CACHE BOOL "" FORCE)
set(USE_OMP TRUE CACHE BOOL "" FORCE)

# add subdirectory gege
add_subdirectory(gege)

# set dendelion target
add_custom_target(gege ALL DEPENDS bindings)

# package and install pip
add_custom_target(pip-install
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/gege/src/python ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/README.md ${PACKAGE_OUTPUT_DIRECTORY}/README.md
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/package/ ${PACKAGE_OUTPUT_DIRECTORY}
    COMMAND ${Python3_EXECUTABLE} -m pip install --no-build-isolation --no-deps .
    WORKING_DIRECTORY ${PACKAGE_OUTPUT_DIRECTORY}
    DEPENDS gege
)
