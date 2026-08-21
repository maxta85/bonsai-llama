cmake_minimum_required(VERSION 3.18)
project(bonsai_llama LANGUAGES CXX)

# Top-level: build the C++ kernel library + CLI, and optionally the bundled
# llama.cpp fork (run scripts/setup_llamacpp.sh first to fetch it).
add_subdirectory(cpp)

option(BONSAI_BUILD_LLAMACPP "Build the bundled PrismML-Eng/llama.cpp fork" OFF)
if(BONSAI_BUILD_LLAMACPP)
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt)
        message(FATAL_ERROR "llama.cpp fork not found. Run scripts/setup_llamacpp.sh first.")
    endif()
    add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)
endif()
