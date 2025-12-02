#include <stdio.h>
#include <cuda_runtime.h>

// Simple CUDA kernel for vector addition
// Each thread computes one element of the result vector
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check to ensure we don't access out of bounds
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    // Vector size
    const int N = 1024;
    const size_t bytes = N * sizeof(float);
    
    // Host vectors
    float *h_a, *h_b, *h_c;
    
    // Device vectors
    float *d_a, *d_b, *d_c;
    
    // Allocate host memory
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
    
    // Initialize host vectors
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }
    
    // Allocate device memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // Copy data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    // Verify results
    printf("Verifying results...\n");
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("Error at index %d: %f != %f\n", i, h_c[i], h_a[i] + h_b[i]);
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("Success! All %d elements computed correctly.\n", N);
        printf("Example: %.1f + %.1f = %.1f\n", h_a[0], h_b[0], h_c[0]);
        printf("Example: %.1f + %.1f = %.1f\n", h_a[N-1], h_b[N-1], h_c[N-1]);
    }
    
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}
