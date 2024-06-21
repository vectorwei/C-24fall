#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gmp.h>
#include <float.h>
#define DBL_DIG         15  
// 函数声明
void add(mpf_t result, const mpf_t a, const mpf_t b);
void subtract(mpf_t result, const mpf_t a, const mpf_t b);
void multiply(mpf_t result, const mpf_t a, const mpf_t b);
void divide(mpf_t result, const mpf_t a, const mpf_t b);

int main() {
    // 初始化大数变量
    mpf_set_default_prec(1024); // 设置默认精度
    mpf_t result, operand, value;
    mpf_inits(result, operand, value, NULL);

    printf("Calculator - type 'quit' to exit\n");

    char input[100];

    while (1) {
        printf("> ");
        if (fgets(input, sizeof(input), stdin) == NULL) {
            break; 
}
        // 移除换行符
        input[strcspn(input, "\n")] = '\0';

        // 检查是否是退出命令
        if (strcmp(input, "quit") == 0) {
            break;
        }
        char op;
        double d_operand, d_value;
       if (sscanf(input, "%lf %c %lf", &d_operand, &op, &d_value) == 3) {
            mpf_set_d(operand, d_operand);
            mpf_set_d(value, d_value);
             {
            switch (op) {
                case '+':
                add(result, operand, value);
                gmp_printf("%.*g %c %.*g = %.*g\n", DBL_DIG, mpf_get_d(operand), op, DBL_DIG, mpf_get_d(value), DBL_DIG, mpf_get_d(result));
                break;
                case '-':
                subtract(result, operand, value);
                gmp_printf("%.*g %c %.*g = %.*g\n", DBL_DIG, mpf_get_d(operand), op, DBL_DIG, mpf_get_d(value), DBL_DIG, mpf_get_d(result));
                break;
                case '*':
                multiply(result, operand, value);
                gmp_printf("%.*g %c %.*g = %.*g\n", DBL_DIG, mpf_get_d(operand), op, DBL_DIG, mpf_get_d(value), DBL_DIG, mpf_get_d(result));
                break;
                case '/':
                    if (mpf_cmp_ui(value, 0) != 0) {
                        divide(result, operand, value);
                        gmp_printf("%.*g %c %.*g = %.*g\n", DBL_DIG, mpf_get_d(operand), op, DBL_DIG, mpf_get_d(value), DBL_DIG, mpf_get_d(result));} 
                    else {printf("Error: Division by zero is not allowed.\n");}
                    break;
                default:
                    printf("Error: Invalid operator '%c'\n", op);
                    break;
            }
        } }
        else {
            printf("Error: Invalid input format.\n");
        }
    }

    // 释放资源
    mpf_clears(result, operand, value, NULL);

    printf("Exiting calculator.\n");

    return 0;
}

// 大数浮点数加法
void add(mpf_t result, const mpf_t a, const mpf_t b) {
    mpf_add(result, a, b);
}

// 大数浮点数减法
void subtract(mpf_t result, const mpf_t a, const mpf_t b) {
    mpf_sub(result, a, b);
}

// 大数浮点数乘法
void multiply(mpf_t result, const mpf_t a, const mpf_t b) {
    mpf_mul(result, a, b);
}

// 大数浮点数除法
void divide(mpf_t result, const mpf_t a, const mpf_t b) {
    mpf_div(result, a, b);
}
