cmake_minimum_required(VERSION 3.18)
project(gpu-native-room-inference LANGUAGES CXX CUDA)

# CUDA configuration
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA architecture (Jetson Orin Nano: sm_87, RTX 4050: sm_89)
set(CUDA_ARCH "87" CACHE STRING "CUDA architecture (87 for Jetson, 89 for RTX 4050)")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Library target
add_library(gpu_native_room_inference STATIC
    kernels/thread_as_room.cu
    kernels/warp_as_room_basic.cu
)

# Benchmark executable
add_executable(benchmark
    benchmarks/benchmark_main.cu
)
target_link_libraries(benchmark gpu_native_room_inference)

# Test executable
add_executable(test_correctness
    tests/test_correctness.cu
)
target_link_libraries(test_correctness gpu_native_room_inference)

# Installation
install(TARGETS gpu_native_room_inference
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/gpu-native-room-inference-config-version.cmake
    VERSION 0.1.0
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gpu-native-room-inference-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/gpu-native-room-inference-config.cmake
    INSTALL_DESTINATION lib/cmake/gpu-native-room-inference
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/gpu-native-room-inference-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/gpu-native-room-inference-config-version.cmake
    DESTINATION lib/cmake/gpu-native-room-inference
)

# Variants for different application domains
add_subdirectory(variants/edge_ai)
add_subdirectory(variants/cloud_serving)

# Variant benchmarks
add_executable(benchmark_variants
    benchmarks/variants/benchmark_variants_main.cu
)
target_link_libraries(benchmark_variants
    gpu_native_room_inference
    edge_ai_variant
    cloud_serving_variant
)

# Additional variants
add_subdirectory(variants/scientific_sim)
add_subdirectory(variants/game_ai)

# Example applications
add_executable(example_edge_ai
    examples/variants/example_edge_ai.cu
)
target_link_libraries(example_edge_ai
    gpu_native_room_inference
    edge_ai_variant
)
