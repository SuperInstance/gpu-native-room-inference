// example_edge_ai.cu - Edge AI example using lightweight warp variant

#include <iostream>
#include <cuda_runtime.h>
#include "room_types.h"

// Forward declarations
extern "C" void launch_warp_lightweight(
    const half* room_inputs, half* room_outputs, const half* weights,
    int num_rooms, int input_dim, int output_dim, cudaStream_t stream);

int main() {
    std::cout << "Edge AI Example: Lightweight Warp Variant" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // Edge AI configuration
    const int NUM_ROOMS = 32;      // Typical edge batch size
    const int INPUT_DIM = 128;     // Edge-optimized input size
    const int OUTPUT_DIM = 64;     // Edge-optimized output size
    
    // Allocate host memory
    size_t input_size = NUM_ROOMS * INPUT_DIM * sizeof(half);
    size_t output_size = NUM_ROOMS * OUTPUT_DIM * sizeof(half);
    size_t weights_size = INPUT_DIM * OUTPUT_DIM * sizeof(half);
    
    half* h_inputs = new half[NUM_ROOMS * INPUT_DIM];
    half* h_outputs = new half[NUM_ROOMS * OUTPUT_DIM];
    half* h_weights = new half[INPUT_DIM * OUTPUT_DIM];
    
    // Initialize with simple pattern (edge devices often have simple inputs)
    for (int i = 0; i < NUM_ROOMS * INPUT_DIM; ++i) {
        h_inputs[i] = __float2half((i % 10) * 0.1f);
    }
    
    for (int i = 0; i < INPUT_DIM * OUTPUT_DIM; ++i) {
        h_weights[i] = __float2half((i % 5) * 0.2f);
    }
    
    // Allocate device memory
    half *d_inputs, *d_outputs, *d_weights;
    cudaMalloc(&d_inputs, input_size);
    cudaMalloc(&d_outputs, output_size);
    cudaMalloc(&d_weights, weights_size);
    
    // Copy to device
    cudaMemcpy(d_inputs, h_inputs, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, weights_size, cudaMemcpyHostToDevice);
    
    // Initialize CUDA events for timing (edge performance measurement)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Run inference
    cudaEventRecord(start);
    
    launch_warp_lightweight(d_inputs, d_outputs, d_weights,
                           NUM_ROOMS, INPUT_DIM, OUTPUT_DIM);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Copy results back
    cudaMemcpy(h_outputs, d_outputs, output_size, cudaMemcpyDeviceToHost);
    
    // Calculate performance
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    float latency_ms = milliseconds;
    float throughput_qps = NUM_ROOMS / (milliseconds / 1000.0f);
    
    std::cout << "\nEdge AI Performance:" << std::endl;
    std::cout << "  Rooms: " << NUM_ROOMS << std::endl;
    std::cout << "  Input dimension: " << INPUT_DIM << std::endl;
    std::cout << "  Output dimension: " << OUTPUT_DIM << std::endl;
    std::cout << "  Latency: " << latency_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << throughput_qps << " qps" << std::endl;
    
    // Check memory usage (important for edge)
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    float memory_mb = (input_size + output_size + weights_size) / (1024.0f * 1024.0f);
    std::cout << "  Memory usage: " << memory_mb << " MB" << std::endl;
    std::cout << "  GPU memory free: " << free_mem / (1024 * 1024) << " MB" << std::endl;
    
    // Verify some results
    std::cout << "\nSample results (first room, first 3 outputs):" << std::endl;
    for (int i = 0; i < 3; ++i) {
        std::cout << "  Output[" << i << "]: " << __half2float(h_outputs[i]) << std::endl;
    }
    
    // Cleanup
    delete[] h_inputs;
    delete[] h_outputs;
    delete[] h_weights;
    cudaFree(d_inputs);
    cudaFree(d_outputs);
    cudaFree(d_weights);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    std::cout << "\nEdge AI example completed successfully." << std::endl;
    return 0;
}
