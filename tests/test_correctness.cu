// test_correctness.cu - Correctness tests for GPU-native room inference

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include "room_types.h"

// Forward declarations from kernels
extern "C" void launch_thread_as_room(
    const half* room_inputs, half* room_outputs, const half* weights,
    int num_rooms, int input_dim, int output_dim, cudaStream_t stream);

extern "C" void launch_warp_as_room_basic(
    const half* room_inputs, half* room_outputs, const half* weights,
    int num_rooms, int input_dim, int output_dim, cudaStream_t stream);

bool compare_results(const half* a, const half* b, int n, float tolerance = 1e-3f) {
    for (int i = 0; i < n; ++i) {
        float fa = __half2float(a[i]);
        float fb = __half2float(b[i]);
        if (fabs(fa - fb) > tolerance) {
            std::cout << "Mismatch at index " << i << ": " << fa << " != " << fb << std::endl;
            return false;
        }
    }
    return true;
}

void test_thread_as_room() {
    std::cout << "Testing thread_as_room kernel..." << std::endl;
    
    const int num_rooms = 32;
    const int input_dim = 256;
    const int output_dim = 128;
    
    // Allocate host memory
    size_t input_size = num_rooms * input_dim * sizeof(half);
    size_t output_size = num_rooms * output_dim * sizeof(half);
    size_t weights_size = input_dim * output_dim * sizeof(half);
    
    half* h_inputs = new half[num_rooms * input_dim];
    half* h_outputs = new half[num_rooms * output_dim];
    half* h_weights = new half[input_dim * output_dim];
    half* h_expected = new half[num_rooms * output_dim];
    
    // Initialize with simple pattern
    for (int i = 0; i < num_rooms * input_dim; ++i) {
        h_inputs[i] = __float2half((i % 10) * 0.1f);
    }
    
    for (int i = 0; i < input_dim * output_dim; ++i) {
        h_weights[i] = __float2half((i % 5) * 0.2f);
    }
    
    // Compute expected results on CPU (simplified)
    for (int r = 0; r < num_rooms; ++r) {
        for (int o = 0; o < output_dim; ++o) {
            float sum = 0.0f;
            for (int i = 0; i < input_dim; ++i) {
                float input = __half2float(h_inputs[r * input_dim + i]);
                float weight = __half2float(h_weights[i * output_dim + o]);
                sum += input * weight;
            }
            h_expected[r * output_dim + o] = __float2half(sum > 0 ? sum : 0);
        }
    }
    
    // Allocate device memory
    half *d_inputs, *d_outputs, *d_weights;
    cudaMalloc(&d_inputs, input_size);
    cudaMalloc(&d_outputs, output_size);
    cudaMalloc(&d_weights, weights_size);
    
    // Copy to device
    cudaMemcpy(d_inputs, h_inputs, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, weights_size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    launch_thread_as_room(d_inputs, d_outputs, d_weights, num_rooms, input_dim, output_dim);
    
    // Copy back
    cudaMemcpy(h_outputs, d_outputs, output_size, cudaMemcpyDeviceToHost);
    
    // Compare
    bool passed = compare_results(h_outputs, h_expected, num_rooms * output_dim);
    
    if (passed) {
        std::cout << "  ✅ thread_as_room test PASSED" << std::endl;
    } else {
        std::cout << "  ❌ thread_as_room test FAILED" << std::endl;
    }
    
    // Cleanup
    delete[] h_inputs;
    delete[] h_outputs;
    delete[] h_weights;
    delete[] h_expected;
    cudaFree(d_inputs);
    cudaFree(d_outputs);
    cudaFree(d_weights);
}

void test_warp_as_room_basic() {
    std::cout << "Testing warp_as_room_basic kernel..." << std::endl;
    
    // Similar test structure
    // For now, just indicate it would be implemented
    std::cout << "  ⏳ warp_as_room_basic test (to be implemented)" << std::endl;
}

int main() {
    std::cout << "GPU-Native Room Inference Correctness Tests" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    test_thread_as_room();
    test_warp_as_room_basic();
    
    std::cout << "\nAll tests completed." << std::endl;
    return 0;
}
