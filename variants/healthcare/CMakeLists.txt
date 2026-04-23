# Healthcare variant (secure warp)
add_library(healthcare_variant STATIC
    warp_secure.cu
)

target_include_directories(healthcare_variant
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(healthcare_variant
    gpu_native_room_inference
)

# Healthcare-specific definitions
target_compile_definitions(healthcare_variant
    PRIVATE 
        PRIVACY_NOISE_SCALE=0.01
        HIPAA_COMPLIANCE_LEVEL=2
        MAX_DIAGNOSIS_CONFIDENCE=0.9
)
