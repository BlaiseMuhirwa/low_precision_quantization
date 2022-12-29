
cmake_minimum_required(VERSION 3.18)

project(low_precision_quantizer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX17_STANDARD_COMPILE_OPTION "-std=c++17")
set(CMAKE_CXX17_EXTENSION_COMPILE_OPTION "-std=gnu++17")

set(SKIP_PERFORMANCE_COMPARISON ON)

## Add compiler flags that should be used 
# - Wall : enable all warnings
# - O3: enable all optimizations
set(CMAKE_CXX_FLAGS "-O3 -Wall -std=c++17")


# Specify the non-header source files that should be compiled
file(GLOB CPP_SOURCE 
    "${PROJECT_SOURCE_DIR}/src/*.cc"
    # "${PROJECT_SOURCE_DIR}/dependencies/cereal"
)
include_directories(${PROJECT_SOURCE_DIR}/src/*.cc)
include_directories(${PROJECT_SOURCE_DIR}/dependencies/cereal)

# where to find header files
include_directories(${PROJECT_SOURCE_DIR}/src)

# Name of the resulting executable, and which source files to 
# use as specified by the CPP_SOURCE variable
add_executable(main ${CPP_SOURCE})

message("===============================================")
message("Project Directory = ${PROJECT_SOURCE_DIR}")
message("C++ Complier = ${CMAKE_CXX_COMPILER}")
message("Source Files = ${CPP_SOURCE}")


# add_subdirectory(dependencies/cereal)

# target_link_libraries(low_precision_quantization cereal::cereal)


