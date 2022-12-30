
cmake_minimum_required(VERSION 3.18)

project(low_precision_quantizer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_COMPILER "/usr/local/opt/llvm/bin")
set(CMAKE_CXX17_STANDARD_COMPILE_OPTION "-std=c++17")
set(CMAKE_CXX17_EXTENSION_COMPILE_OPTION "-std=gnu++17")

set(SKIP_PERFORMANCE_COMPARISON ON)

find_package(OpenMP)
if(NOT OpenMP_FOUND)
    message("OpenMP Not Found.")
    set(LLVM_OPENMP_VERSION "13.0.1")
    set(LLVM_OPENMP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/openmp")

    if(NOT EXISTS ${LLVM_OPENMP_SOURCE_DIR})
        message("Downloading openmp-${LLVM_OPENMP_VERSION} into ${LLVM_OPENMP_SOURCE_DIR}")
        set(LLVM_OPENMP_DOWNLOAD_URL "https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_OPENMP_VERSION}/openmp-${LLVM_OPENMP_VERSION}.src.tar.xz")

        file(
            DOWNLOAD
            ${LLVM_OPENMP_DOWNLOAD_URL}
            "${CMAKE_CURRENT_BINARY_DIR}/openmp-${LLVM_OPENMP_VERSION}.tar.bz2"
        )
        file(ARCHIVE_EXTRACT INPUT
            "${CMAKE_CURRENT_BINARY_DIR}/openmp-${LLVM_OPENMP_VERSION}.tar.bz2"
            DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
        )
        file(RENAME
            "${CMAKE_CURRENT_BINARY_DIR}/openmp-${LLVM_OPENMP_VERSION}.src"
            "${LLVM_OPENMP_SOURCE_DIR}"
        )
    endif()
endif()

## Add compiler flags that should be used 
# - Wall : enable all warnings
# - O3: enable all optimizations
set(CMAKE_CXX_FLAGS "-O3 -Wall -fopenmp -std=c++17")


# Specify the non-header source files that should be compiled
file(GLOB CPP_SOURCE 
    "${PROJECT_SOURCE_DIR}/src/*.cc"
    # "${PROJECT_SOURCE_DIR}/dependencies/cereal"
)

# This is so we can use the include macro 
include_directories(".")
include_directories(${PROJECT_SOURCE_DIR}/src/*.cc)
# include_directories(${PROJECT_SOURCE_DIR}/dependencies/cereal)

# where to find header files
include_directories(${PROJECT_SOURCE_DIR}/src)

# Name of the resulting executable, and which source files to 
# use as specified by the CPP_SOURCE variable
add_executable(main ${CPP_SOURCE})

message("===============================================")
message("\tProject Directory = ${PROJECT_SOURCE_DIR}")
message("\tC++ Complier = ${CMAKE_CXX_COMPILER}")
message("\tSource Files = ${CPP_SOURCE}")


# add_subdirectory(dependencies/cereal)

# target_link_libraries(low_precision_quantization cereal::cereal)


