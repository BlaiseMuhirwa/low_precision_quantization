
enable_testing()

include(GoogleTest)

add_executable(LPQTest TestQuantizer.cc)
add_executable(ExactSearchTest TestExactSearch.cc)

target_link_libraries(LPQTest gtest gtest_main lpq)
target_link_libraries(ExactSearchTest gtest gtest_main lpq)


