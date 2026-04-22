# Scientific simulation variant (intelligent warp)
add_library(scientific_sim_variant STATIC
    warp_intelligent.cu
)

target_include_directories(scientific_sim_variant
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(scientific_sim_variant
    gpu_native_room_inference
)

# Requires cooperative groups (CUDA 9+)
target_compile_definitions(scientific_sim_variant
    PRIVATE USE_COOPERATIVE_GROUPS=1
)
