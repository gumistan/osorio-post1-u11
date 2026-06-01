#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE 16

__global__ void matMulTiled(const float *A, const float *B,
                             float *C, int N) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
        if (row < N && t*TILE + threadIdx.x < N)
            sA[threadIdx.y][threadIdx.x] = A[row*N + t*TILE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t*TILE + threadIdx.y < N)
            sB[threadIdx.y][threadIdx.x] = B[(t*TILE + threadIdx.y)*N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }
    if (row < N && col < N) C[row*N + col] = sum;
}

__global__ void matMulNaive(const float *A, const float *B,
                             float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row*N + k] * B[k*N + col];
        C[row*N + col] = sum;
    }
}

void matMulCPU(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
                sum += A[i*N + k] * B[k*N + j];
            C[i*N + j] = sum;
        }
}

void benchmark(int N) {
    size_t bytes = N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_cpu = (float*)malloc(bytes);
    float *h_C_gpu = (float*)malloc(bytes);

    for (int i = 0; i < N*N; i++) {
        h_A[i] = (float)(i % 100) * 0.01f;
        h_B[i] = (float)(i % 100) * 0.02f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((N+TILE-1)/TILE, (N+TILE-1)/TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0;

    cudaEventRecord(start);
    matMulNaive<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("N=%d | Naive GPU: %.2f ms\n", N, ms);

    cudaEventRecord(start);
    matMulTiled<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("N=%d | Tiled GPU: %.2f ms\n", N, ms);

    cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);

    if (N <= 512) {
        matMulCPU(h_A, h_B, h_C_cpu, N);
        int errors = 0;
        for (int i = 0; i < N*N; i++)
            if (fabs(h_C_gpu[i] - h_C_cpu[i]) > 1e-3f) errors++;
        printf("N=%d | Errores: %d\n", N, errors);
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
}

int main() {
    benchmark(512);
    benchmark(1024);
    return 0;
}
