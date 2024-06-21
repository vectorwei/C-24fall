#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error calling \"" #call "\", code is " << err << std::endl; \
        exit(-1); \
    } \
}
void initializeMatrix(float* matrix, int N) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    const int N = 4096;
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;

    A = new float[N * N];
    B = new float[N * N];
    C = new float[N * N];

    initializeMatrix(A, N);
    initializeMatrix(B, N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void **)&d_A, N * N * sizeof(float));
    cudaMalloc((void **)&d_B, N * N * sizeof(float));
    cudaMalloc((void **)&d_C, N * N * sizeof(float));

    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;
    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
    cudaEventRecord(stop);

    CUDA_CHECK(cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU Multiplication Time: " << milliseconds << " ms." << std::endl;

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
