cmake_minimum_required(VERSION 3.18)
project(bonsai_kernels_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(test_kernels test_kernels.cpp)
target_link_libraries(test_kernels PRIVATE bonsai_kernels)
add_test(NAME bonsai_kernels COMMAND test_kernels)
