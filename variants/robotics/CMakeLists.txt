# Robotics variant (deterministic real-time warp)
add_library(robotics_variant STATIC
    warp_deterministic.cu
)

target_include_directories(robotics_variant
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(robotics_variant
    gpu_native_room_inference
)

# Robotics-specific definitions
target_compile_definitions(robotics_variant
    PRIVATE 
        MAX_EXECUTION_TIME_US=100
        SAFETY_LIMIT_MIN=-1.5
        SAFETY_LIMIT_MAX=1.5
)
