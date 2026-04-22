# Edge AI variant (lightweight warp)
add_library(edge_ai_variant STATIC
    warp_lightweight.cu
)

target_include_directories(edge_ai_variant
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(edge_ai_variant
    gpu_native_room_inference
)
