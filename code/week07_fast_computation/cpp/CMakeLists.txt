cmake_minimum_required(VERSION 3.15)
project(fastmm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# 1) Standalone demo -- needs no Python, always builds.
add_executable(cos_demo main.cpp cos.cpp)

# 2) Python module via pybind11 (optional; builds if pybind11 is found).
#    Configure with:  -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
  pybind11_add_module(fastmm bindings.cpp cos.cpp)
  message(STATUS "pybind11 found -- building the 'fastmm' Python module.")
else()
  message(WARNING "pybind11 not found -- building 'cos_demo' only. "
                  "Re-run with -Dpybind11_DIR=$(python -m pybind11 --cmakedir) "
                  "to also build the Python module.")
endif()
