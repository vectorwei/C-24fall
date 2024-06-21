
#include <cblas.h>
#include <iostream>
#include <sys/time.h>
void initializeMatrix(float* matrix, int N) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}
int main() {
    const int N = 4096;  
    float *A = new float[N * N];
    float *B = new float[N * N];
    float *C = new float[N * N];
    initializeMatrix(A, N);
    initializeMatrix(B, N);

    struct timeval start, end;
    gettimeofday(&start, NULL);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 0.0, C, N);
    gettimeofday(&end, NULL);
    double elapsed_time = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
    std::cout << "CPU Multiplication Time: " << elapsed_time << " ms" << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
