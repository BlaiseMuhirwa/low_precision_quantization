cmake_minimum_required(VERSION 3.4...3.18)

project(low_precision_quantizer LANGUAGES CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# set(CMAKE_CXX_COMPILER "/usr/local/opt/llvm/bin")


find_package(Python COMPONENTS Interpreter Development)

include_directories(".")
include_directories(${CMAKE_BINARY_DIR})
include_directories(dependencies)
add_subdirectory(src)


# Add source dependencies 
add_subdirectory(dependencies/pybind11 EXCLUDE_FROM_ALL)
add_subdirectory(dependencies/googletest EXCLUDE_FROM_ALL)

# set(SKP_PERFORMANCE_COMPARISON ON)
# add_subdirectory(dependencies/cereal EXCLUDE_FROM_ALL)

# Add OpenMP
find_package(OpenMP REQUIRED)
if(NOT OpenMP_FOUND)
    message("OpenMP Not Found. Building the Package using LLVM's OpenMP.")
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
    set(OPENMP_STANDALONE_BUILD ON)
    add_subdirectory(${LLVM_OPENMP_SOURCE_DIR} EXCLUDE_FROM_ALL)
    add_library(OpenMP::OpenMP_CXX ALIAS omp)

endif(NOT OpenMP_FOUND)


# Add compiler flags that should be used - Wall : enable all warnings - O3:
# enable all optimizations
set(CMAKE_CXX_FLAGS "-O3 -W")

# Specify the non-header source files that should be compiled
file(GLOB CPP_SOURCE "${PROJECT_SOURCE_DIR}/src/*.cc"
     # "${PROJECT_SOURCE_DIR}/dependencies/cereal"
)
# where to find header files
include_directories(${PROJECT_SOURCE_DIR}/src)

add_executable(main ${CPP_SOURCE})

set(LPQ_SOURCES
    ${PROJECT_SOURCE_DIR}/src/LPQ.cc
    ${PROJECT_SOURCE_DIR}/src/ExactSearch.cc
    ${PROJECT_SOURCE_DIR}/src/NaiveQuantizer.cc)
add_library(_lpq STATIC ${LPQ_SOURCES})
set_target_properties(_lpq PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(_lpq PUBLIC OpenMP::OpenMP_CXX)


pybind11_add_module(lpq ${PROJECT_SOURCE_DIR}/bindings/PythonLPQ.cc)
target_link_libraries(lpq PUBLIC _lpq)



message("===================================================")
message("\tProject Directory = ${PROJECT_SOURCE_DIR}")
message("\tC++ Compiler = ${CMAKE_CXX_COMPILER}")
message("\tSource Files = ")
foreach(SOURCE_FILE IN LISTS CPP_SOURCE)
    message("\t ${SOURCE_FILE}")
endforeach(SOURCE_FILE IN LISTS CPP_SOURCE)


