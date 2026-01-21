cmake_minimum_required(VERSION 3.20)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

project(Echo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(HDF5 REQUIRED COMPONENTS C)

add_executable(echo
    src/main.cpp
    src/Simulate.cpp
    src/Trajectory.cpp
    src/GaseousDynamics.cpp
    src/RootFinder.cpp
    src/CheckpointWriter.cpp
)

target_include_directories(echo PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(echo PRIVATE
    ${HDF5_LIBRARIES}
)

target_compile_options(echo PRIVATE
  $<$<CONFIG:Release>:
    -mcpu=native
    -ffp-contract=fast
    -fno-math-errno
    -fno-trapping-math
  >
)

set_property(TARGET echo PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)


