
enable_testing()

include(GoogleTest)

add_executable(LPQTest test.cc)
target_link_libraries(LPQTest gtest gtest_main lpq)
