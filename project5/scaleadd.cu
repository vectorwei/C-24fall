#include <stdio.h>
#include <sys/time.h>

#define TIME_START gettimeofday(&t_start, NULL);
#define TIME_END(name) gettimeofday(&t_end, NULL); \
                      elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0; \
                      elapsedTime += (t_end.tv_usec - t_start.tv_usec) / 1000.0; \
                      printf(#name " Time = %f ms\n", elapsedTime);

typedef struct {
    size_t rows;
    size_t cols;
    float *data;
    float *data_device;
} Matrix;

Matrix *initializeMatrix(size_t r, size_t c) {
    size_t len = r * c;
    if (len == 0) {
        fprintf(stderr, "Invalid size. The input should be > 0.\n");
        return NULL;
    }
    Matrix *p = (Matrix *)malloc(sizeof(Matrix));
    if (p == NULL) {
        fprintf(stderr, "Allocate host memory failed.\n");
        goto ERR_TAG;
    }
    p->rows = r;
    p->cols = c;
    p->data = (float *)malloc(sizeof(float) * len);
    if (p->data == NULL) {
        fprintf(stderr, "Allocate host memory failed.\n");
        goto ERR_TAG;
    }
    if (cudaMalloc(&p->data_device, sizeof(float) * len) != cudaSuccess) {
        fprintf(stderr, "Allocate device memory failed.\n");
        goto ERR_TAG;
    }
    return p;
ERR_TAG:
    if (p && p->data) free(p->data);
    if (p) free(p);
    return NULL;
}

void freeMatrix(Matrix **pp) {
    if (pp == NULL) return;
    Matrix *p = *pp;
    if (p != NULL) {
        if (p->data) free(p->data);
        if (p->data_device) cudaFree(p->data_device);
    }
    *pp = NULL;
}

bool setMatrix(Matrix *pMat, float val) {
    if (pMat == NULL) {
        fprintf(stderr, "NULL pointer.\n");
        return false;
    }
    size_t len = pMat->rows * pMat->cols;
    for (size_t i = 0; i < len; i++)
        pMat->data[i] = val;

    return true;
}

__global__ void scaleAddKernel(const float *input, float *output, size_t len, float a, float b) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < len)
        output[i] = a * input[i] + b;
}

bool scaleAddGPU(const Matrix *pMat, Matrix *pResult, float a, float b) {
    if (pMat == NULL || pResult == NULL) {
        fprintf(stderr, "Null pointer.\n");
        return false;
    }
    if (pMat->rows != pResult->rows || pMat->cols != pResult->cols) {
        fprintf(stderr, "The matrices are not of the same size.\n");
        return false;
    }

    cudaError_t ecode = cudaSuccess;
    size_t len = pMat->rows * pMat->cols;

    cudaMemcpy(pMat->data_device, pMat->data, sizeof(float) * len, cudaMemcpyHostToDevice);
    scaleAddKernel<<<(len + 255) / 256, 256>>>(pMat->data_device, pResult->data_device, len, a, b);
    if ((ecode = cudaGetLastError()) != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(ecode));
        return false;
    }
    cudaMemcpy(pResult->data, pResult->data_device, sizeof(float) * len, cudaMemcpyDeviceToHost);

    return true;
}

bool scaleAddCPU(const Matrix *pMat, Matrix *pResult, float a, float b) {
    if (pMat == NULL || pResult == NULL) {
        fprintf(stderr, "Null pointer.\n");
        return false;
    }
    if (pMat->rows != pResult->rows || pMat->cols != pResult->cols) {
        fprintf(stderr, "Not of the same size.\n");
        return false;
    }
    size_t len = pMat->rows * pMat->cols;
    for (size_t i = 0; i < len; i++) {
        pResult->data[i] = a * pMat->data[i] + b;
    }
    return true;
}

int main() {
    struct timeval t_start, t_end;
    double elapsedTime = 0;

    int dev_count = 0;
    int dev_id = 0;
    cudaGetDeviceCount(&dev_count);
    cudaSetDevice(2);  // Ensure to set to an existing device ID
    cudaGetDevice(&dev_id);
    printf("You have %d cuda devices.\n", dev_count);
    printf("You are using device %d.\n", dev_id);

    float scalarA = 5.5;  // Example value for a
    float scalarB = 12.3;  // Example value for b

    Matrix *pMat1 = initializeMatrix(4096, 4096);
    Matrix *pResult = initializeMatrix(4096, 4096);

    setMatrix(pMat1, 1.1f);

    TIME_START
    scaleAddCPU(pMat1, pResult, scalarA, scalarB);
    TIME_END(scaleAddCPU)
    printf("  Result = [%.1f, ..., %.1f]\n", pResult->data[0], pResult->data[pResult->rows * pResult->cols - 1]);

    TIME_START
    scaleAddGPU(pMat1, pResult, scalarA, scalarB);
    TIME_END(scaleAddGPU)
    printf("  Result = [%.1f, ..., %.1f]\n", pResult->data[0], pResult->data[pResult->rows*pResult->cols-1]);

    freeMatrix(&pMat1);
    freeMatrix(&pResult);
    return 0;
}
