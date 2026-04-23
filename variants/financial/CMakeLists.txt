# Financial variant (high-precision warp)
add_library(financial_variant STATIC
    warp_highprecision.cu
)

target_include_directories(financial_variant
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(financial_variant
    gpu_native_room_inference
)

# Financial-specific definitions
target_compile_definitions(financial_variant
    PRIVATE 
        FINANCIAL_PRECISION_THRESHOLD=1e-4
        MAX_POSITION_LIMIT=10000.0
        MAX_RISK_LIMIT=0.95
)
