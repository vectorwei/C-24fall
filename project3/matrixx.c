#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <pmmintrin.h> 
#include <omp.h>       
#include <cblas.h>

#define BLOCK_SIZE 16 

void matmul_openblas(float *input_matrix_A, float *input_matrix_B, float *output_matrix_C, int size) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, size, size, 1.0f,
                input_matrix_A, size, input_matrix_B, size, 0.0f, output_matrix_C, size);
}

void matmul_plain(float *input_matrix_A, float *input_matrix_B, float *output_matrix_C, int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			output_matrix_C[i * size + j] = 0;
			for (int k = 0; k < size; k++) {
				output_matrix_C[i * size + j] += input_matrix_A[i * size + k] * input_matrix_B[k * size + j];
			}
		}
	}
}

void matmul_improved(float *input_matrix_A, float *input_matrix_B, float *output_matrix_C, int size) {
    int i, j, k, i0, j0, k0;
    float *pA, *pB, sum;
    __m128 a_line, b_line, r_line;

#pragma omp parallel for private(i, j, k, i0, j0, k0, pA, pB, a_line, b_line, r_line, sum) shared(input_matrix_A, input_matrix_B, output_matrix_C)
    for (i = 0; i < size; i += BLOCK_SIZE) {
        for (j = 0; j < size; j += BLOCK_SIZE) {
            for (k = 0; k < size; k += BLOCK_SIZE) {
                for (i0 = i; i0 < i + BLOCK_SIZE; ++i0) {
                    for (j0 = j; j0 < j + BLOCK_SIZE; j0 += 4) {
                        r_line = _mm_setzero_ps();
                        for (k0 = k; k0 < k + BLOCK_SIZE; k0 += 4) {
                            pA = input_matrix_A + i0 * size + k0;
                            pB = input_matrix_B + k0 * size + j0;
                            a_line = _mm_loadu_ps(pA);
                            b_line = _mm_loadu_ps(pB);
                            r_line = _mm_add_ps(r_line, _mm_mul_ps(a_line, _mm_shuffle_ps(b_line, b_line, 0x00)));
                            b_line = _mm_loadu_ps(pB + size);
                            r_line = _mm_add_ps(r_line, _mm_mul_ps(a_line, _mm_shuffle_ps(b_line, b_line, 0x55)));
                            b_line = _mm_loadu_ps(pB + 2*size);
                            r_line = _mm_add_ps(r_line, _mm_mul_ps(a_line, _mm_shuffle_ps(b_line, b_line, 0xAA)));
                            b_line = _mm_loadu_ps(pB + 3*size);
                            r_line = _mm_add_ps(r_line, _mm_mul_ps(a_line, _mm_shuffle_ps(b_line, b_line, 0xFF)));
                        }
                        _mm_storeu_ps(output_matrix_C + i0 * size + j0, r_line);
                    }
                }
            }
        }
    }
}

void generate_random_matrix(float *matrix, int size) {
	for (int i = 0; i < size * size; i++) {
		matrix[i] = (float)rand() / RAND_MAX;
	}
}

int main() {
	int matrix_sizes[] = {16, 128, 256, 512, 1000，2000};

	for (int test = 0; test < 6; test++) {
		int size = matrix_sizes[test];
		float *matrix_A = (float *)malloc(size * size * sizeof(float));
		float *matrix_B = (float *)malloc(size * size * sizeof(float));
		float *matrix_C = (float *)malloc(size * size * sizeof(float));
		
		generate_random_matrix(matrix_A, size);
		generate_random_matrix(matrix_B, size);
		
		clock_t start, end;
		double cpu_time_used;
		
		start = clock();
		matmul_plain(matrix_A, matrix_B, matrix_C, size);
		end = clock();
		cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
		printf("Plain time for %dx%d: %f\n", size, size, cpu_time_used);
		
		start = clock();
		matmul_improved(matrix_A, matrix_B, matrix_C, size);
		end = clock();
		cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
		printf("Improved time for %dx%d: %f\n", size, size, cpu_time_used);

		start = clock();
		matmul_openblas(matrix_A, matrix_B, matrix_C, size);
		end = clock();
		cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
		printf("OpenBLAS time for %dx%d: %f\n", size, size, cpu_time_used);

		free(matrix_A);
		free(matrix_B);
		free(matrix_C);
	}
	
	return 0;
}
