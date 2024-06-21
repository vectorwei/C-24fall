#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 函数声明
long long add(long long a, long long b);
long long subtract(long long a, long long b);
long long multiply(long long a, long long b);
long long divide(long long a, long long b);

int main() {
    printf("Calculator - type 'quit' to exit\n");

    char input[100];
    long long result = 0.0;

    while (1) {
        // 接受用户输入
        printf("> ");
        if (fgets(input, sizeof(input), stdin) == NULL) {
            break; // 如果输入结束，退出循环
        }


        // 移除换行符
        input[strcspn(input, "\n")] = '\0';

        // 检查是否是退出命令
        if (strcmp(input, "quit") == 0) {
            break;
        }

        // 输出用户输入
        printf("%s", input);

        // 解析输入并进行计算
        char op;
        long long operand;
        if (sscanf(input, "%lld %c %lld", &result, &op, &operand) == 3) {
            switch (op) {
                case '+':
                    result = add(result, operand);
                    printf(" = %lld\n", result);
                    break;
                case '-':
                    result = subtract(result, operand);
                    printf(" = %lld\n", result);
                    break;
                case '*':
                    result = multiply(result, operand);
                    printf(" = %lld\n", result);
                    break;
                case '/':
                    if (operand != 0) {
                        result = divide(result, operand);
                    } else {
                        printf("Error: Division by zero is not allowed.\n");
                    }
                    printf(" = %lld\n", result);
                    break;
                default:
                    printf("Error: Invalid operator '%c'\n", op);
                    break;
            }

            // 输出计算结果
            
        } else {
            printf("Error: Invalid input format.\n");
        }
    }

    printf("Exiting calculator.\n");

    return 0;
}

// 加法
long long add(long long a, long long b) {
    return a + b;
}

// 减法
long long subtract(long long a, long long b) {
    return a - b;
}

// 乘法
long long multiply(long long a,long long b) {
    return a * b;
}

// 除法
long long divide(long long a, long long b) {
    return a / b;
}
