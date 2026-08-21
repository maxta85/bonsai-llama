cmake_minimum_required(VERSION 3.18)
project(bonsai_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BONSAI_BUILD_CLI "Build the standalone inference CLI" ON)

add_library(bonsai_kernels STATIC
    kernels/q1_0.cpp
    kernels/q2_0.cpp
)
target_include_directories(bonsai_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/kernels)

if(BONSAI_BUILD_CLI)
    add_executable(bonsai_infer cli/bonsai_infer.cpp)
    target_link_libraries(bonsai_infer PRIVATE bonsai_kernels)
endif()

enable_testing()
add_subdirectory(tests)
