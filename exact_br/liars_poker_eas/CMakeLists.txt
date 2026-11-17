cmake_minimum_required(VERSION 3.20)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O1 -g -fno-omit-frame-pointer")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -DNDEBUG -O3 -march=native")
endif()

python_add_library(pylpeas MODULE csrc/bindings.cpp csrc/lp_tree.cpp WITH_SOABI)
target_link_libraries(pylpeas PRIVATE pybind11::headers OpenMP::OpenMP_CXX)
install(TARGETS pylpeas DESTINATION ${SKBUILD_PROJECT_NAME})
