#include <stdio.h>

#include <time.h>
#include <stdlib.h>
#include <pmmintrin.h> 
#include <omp.h>       // 包含OpenMP头文件
#include <cblas.h>

#define BLOCK_SIZE 16 // 应基于你的CPU缓存大小进行调整
//OpenBLAS进行矩阵乘法：
void matmul_openblas(float *A, float *B, float *C, int N) {
    // OpenBLAS使用列主序存储，所以传递参数时要注意转置
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0f,
                A, N, B, N, 0.0f, C, N);
}

// 简单的矩阵乘法实现
void matmul_plain(float *A, float *B, float *C, int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			C[i * N + j] = 0;
			for (int k = 0; k < N; k++) {
				C[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
}


// 更改的矩阵乘法实现
void matmul_improved(float *A, float *B, float *C, int N) {
    int i, j, k, i0, j0, k0;
    float *pA, *pB, sum;
    __m128 a_line, b_line, r_line;

#pragma omp parallel for private(i, j, k, i0, j0, k0, pA, pB, a_line, b_line, r_line, sum) shared(A, B, C)
    for (i = 0; i < N; i += BLOCK_SIZE) {
        for (j = 0; j < N; j += BLOCK_SIZE) {
            for (k = 0; k < N; k += BLOCK_SIZE) {
                for (i0 = i; i0 < i + BLOCK_SIZE; ++i0) {
                    for (j0 = j; j0 < j + BLOCK_SIZE; j0 += 4) {
                        r_line = _mm_setzero_ps();
                        for (k0 = k; k0 < k + BLOCK_SIZE; k0 += 4) {
                            pA = A + i0 * N + k0;
                            pB = B + k0 * N + j0;
                            a_line = _mm_loadu_ps(pA);
                            b_line = _mm_loadu_ps(pB);
                            r_line = _mm_add_ps(r_line, _mm_mul_ps(a_line, _mm_shuffle_ps(b_line, b_line, 0x00)));
                            b_line = _mm_loadu_ps(pB + N);
                            r_line = _mm_add_ps(r_line, _mm_mul_ps(a_line, _mm_shuffle_ps(b_line, b_line, 0x55)));
                            b_line = _mm_loadu_ps(pB + 2*N);
                            r_line = _mm_add_ps(r_line, _mm_mul_ps(a_line, _mm_shuffle_ps(b_line, b_line, 0xAA)));
                            b_line = _mm_loadu_ps(pB + 3*N);
                            r_line = _mm_add_ps(r_line, _mm_mul_ps(a_line, _mm_shuffle_ps(b_line, b_line, 0xFF)));
                        }
                        _mm_storeu_ps(C + i0 * N + j0, r_line);
                    }
                }
            }
        }
    }
}
// 函数用于生成随机矩阵
void generate_random_matrix(float *mat, int N) {
	for (int i = 0; i < N * N; i++) {
		mat[i] = (float)rand() / RAND_MAX;
	}
}

int main() {
	// 测试不同大小的矩阵
	//int sizes[] = {16, 128, 1024, 8192, 65536};
	int sizes[] = {16, 128,256,512,1024, 2048, 4096};
	for (int test = 0; test < 7; test++) {
		int N = sizes[test];
		float *A = (float *)malloc(N * N * sizeof(float));
		float *B = (float *)malloc(N * N * sizeof(float));
		float *C = (float *)malloc(N * N * sizeof(float));
		
		generate_random_matrix(A, N);
		generate_random_matrix(B, N);
		
		clock_t start, end;
		double cpu_time_used;
		
		// 测试 matmul_plain 计算传统方法时间，觉得跑得慢就把它注释掉 ，只跑下面的优化后的
		start = clock();
		matmul_plain(A, B, C, N);
		end = clock();
		cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
		printf("Plain matrix multiplication time for %dx%d: %f\n", N, N, cpu_time_used);
		
		// 测试 matmul_improved
		start = clock();
		matmul_improved(A, B, C, N);
		end = clock();
		cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
		printf("Improved matrix multiplication time for %dx%d: %f\n", N, N, cpu_time_used);

		// 测试OpenBLAS的矩阵乘法
		start = clock();
		matmul_openblas(A, B, C, N);
		end = clock();
		cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
		printf("OpenBLAS matrix multiplication time for %dx%d: %f\n", N, N, cpu_time_used);

		
		free(A);
		free(B);
		free(C);
	}
	
	return 0;
}

