#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] + B[idx];
}

void vectorAddCPU(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) C[i] = A[i] + B[i];
}

void benchmark(int N, const char* label) {
    size_t bytes = N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_cpu = (float*)malloc(bytes);
    float *h_C_gpu = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i * 0.5f;
        h_B[i] = (float)i * 1.5f;
    }

    // CPU
    clock_t t0 = clock();
    vectorAddCPU(h_A, h_B, h_C_cpu, N);
    double cpu_ms = (double)(clock()-t0)/CLOCKS_PER_SEC*1000.0;

    // GPU
    float *d_A, *d_B, *d_C;
    cudaEvent_t start, stop, k0, k1;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventCreate(&k0); cudaEventCreate(&k1);

    cudaEventRecord(start);
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaEventRecord(k0);
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaEventRecord(k1);

    cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernel_ms = 0, total_ms = 0;
    cudaEventElapsedTime(&kernel_ms, k0, k1);
    cudaEventElapsedTime(&total_ms, start, stop);

    printf("%s | CPU: %.2f ms | GPU kernel: %.2f ms | GPU total: %.2f ms\n",
           label, cpu_ms, kernel_ms, total_ms);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
}

int main() {
    benchmark(1 << 20, "N=1M ");
    benchmark(1 << 22, "N=4M ");
    benchmark(1 << 24, "N=16M");
    return 0;
}
