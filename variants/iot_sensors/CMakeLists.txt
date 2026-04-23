# IoT variant (ultralight warp)
add_library(iot_variant STATIC
    warp_ultralight.cu
)

target_include_directories(iot_variant
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(iot_variant
    gpu_native_room_inference
)

# IoT-specific definitions
target_compile_definitions(iot_variant
    PRIVATE 
        IOT_WARP_SIZE=16
        MAX_READINGS=32
        CRITICAL_BATTERY_LEVEL=0.1
)
