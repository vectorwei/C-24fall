#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h> 
void multiply(float *A, float *B, float *C, int a, int b, int c) {
    for (int i = 0; i < a; ++i) {
        for (int j = 0; j < c; ++j) {
            C[i*c + j] = 0;
            for (int k = 0; k < b; ++k) {
                C[i*c + j] += A[i*b + k] * B[k*c + j];
            }
        }
    }
}


int main() {
    FILE *fp;
    fp = fopen("matrix_timesO3.txt", "w");
    fprintf(fp, "sizeA\tsizeB\ttime\n");
    if (fp == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    char input[20];
    int a, b, c;
    while (1) {
        memset(input, 0, sizeof(input)); // 清空 input 数组
        printf("Enter dimensions (a b c) for square matrix where A is axb and B is bxc (type 'quit' to exit): ");
        fgets(input, sizeof(input), stdin); // 读取一行输入

        // 移除输入字符串末尾的换行符
        input[strcspn(input, "\n")] = '\0';

        if (strcmp(input, "quit") == 0) { // 判断是否输入 "quit"
            break;
        }

        sscanf(input, "%d %d %d", &a, &b, &c); // 解析输入


        float *A = (float *)malloc(a * b * sizeof(float));
        float *B = (float *)malloc(b * c * sizeof(float));
        float *C = (float *)malloc(a * c * sizeof(float));

        srand(time(NULL));
        for (int i = 0; i < a * b; i++) {
            A[i] = rand() / (float)RAND_MAX;
        }
        for (int i = 0; i < b * c; i++) {
            B[i] = rand() / (float)RAND_MAX;
        }

        clock_t start, end;
        double total_time = 0.0;
        for (int i = 0; i < 10; i++) {
            start = clock();
            multiply(A, B, C, a, b, c);
            end = clock();
            total_time += (double)(end - start) / CLOCKS_PER_SEC;
        }

 
        printf("avg time spent for matrix multiplication (C_%dx%d = A_%dx%d * B_%dx%d): %f seconds\n", a, c, a, b, b, c, total_time/10);
        fprintf(fp, "%dx%d\t%dx%d\t%f\n", a, b, b, c, total_time/10);

        free(A);
        free(B);
        free(C);
    }
    fclose(fp);
    return 0;
}