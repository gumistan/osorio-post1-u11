#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1 << 24) // 16M elementos

// Kernel CUDA: cada thread calcula un elemento de C
__global__ void vectorAdd(const float *A, const float *B,
                          float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] + B[idx];
}

// Suma de vectores en CPU (referencia)
void vectorAddCPU(const float *A, const float *B,
                  float *C, int n) {
    for (int i = 0; i < n; i++) C[i] = A[i] + B[i];
}

int main() {
    size_t bytes = N * sizeof(float);

    // Asignar memoria en host
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_cpu = (float*)malloc(bytes);
    float *h_C_gpu = (float*)malloc(bytes);

    // Inicializar vectores
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i * 0.5f;
        h_B[i] = (float)i * 1.5f;
    }

    // --- Benchmark CPU ---
    clock_t t0 = clock();
    vectorAddCPU(h_A, h_B, h_C_cpu, N);
    double cpu_ms = (double)(clock()-t0)/CLOCKS_PER_SEC*1000.0;
    printf("CPU: %.2f ms\n", cpu_ms);

    // --- Benchmark GPU ---
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaEventRecord(start);
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    printf("GPU kernel: %.2f ms\n", gpu_ms);

    cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verificar resultados
    int errors = 0;
    for (int i = 0; i < N; i++)
        if (fabs(h_C_gpu[i] - h_C_cpu[i]) > 1e-4f) errors++;
    printf("Errores: %d\n", errors);

    // Liberar memoria
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    return 0;
}
