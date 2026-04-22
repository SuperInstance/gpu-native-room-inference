# Cloud serving variant (high-throughput warp)
add_library(cloud_serving_variant STATIC
    warp_high_throughput.cu
)

target_include_directories(cloud_serving_variant
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(cloud_serving_variant
    gpu_native_room_inference
)

# Cloud-specific optimizations require CUDA architecture 70+ for tensor cores
if(CMAKE_CUDA_ARCHITECTURES MATCHES "70|75|80|86|87|89|90")
    target_compile_definitions(cloud_serving_variant
        PRIVATE USE_TENSOR_CORES=1
    )
endif()
