# Game AI variant
add_library(game_ai_variant STATIC
    warp_game_ai.cu
)

target_include_directories(game_ai_variant
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(game_ai_variant
    gpu_native_room_inference
)

# Game AI specific definitions
target_compile_definitions(game_ai_variant
    PRIVATE 
        ACTION_DIM=8
        PRIORITY_THRESHOLD=50
)
