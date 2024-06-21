// matrix_multiplication_gpu.cu
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

// CUDA error wrapper
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error calling \"" #call "\", code is " << err << std::endl; \
        exit(-1); \
    } \
}

// Initialize matrix with random data
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
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaMalloc((void **)&d_A, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_B, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_C, N * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;
    CUDA_CHECK(cudaEventRecord(start));
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU Matrix Multiplication Time: " << milliseconds << " ms." << std::endl;

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
