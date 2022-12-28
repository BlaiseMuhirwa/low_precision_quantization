
cmake_minimum_required(VERSION 3.8...3.18)

set(CMAKE_CXX_STANDARD 17)

message("============================")
message("C++ Complier = ${CMAKE_CXX_COMPILER}")

project(low_precision_quantizer LANGUAGES CXX VERSION 1.3.2)

add_subdirectory(src)
add_subdirectory(deps/cereal EXCLUDE_FROM_ALL)


