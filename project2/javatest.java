import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;

public class javatest {
    public static void matrixMultiply(float[][] A, float[][] B, float[][] C) {
        int m = A.length;
        int n = A[0].length;
        int p = B[0].length;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                C[i][j] = 0;
                for (int k = 0; k < n; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }

    public static void main(String[] args) {
        try {
            FileWriter writer = new FileWriter("java_matrix_times.txt");
            writer.write(String.format("sizeA\tsizeB\ttime\n"));

            Scanner scanner = new Scanner(System.in);
            String input;
            while (true) {
                System.out.print("Enter dimensions (m n p) for matrix multiplication (A_mxn * B_nxp) (type 'quit' to exit): ");
                input = scanner.nextLine();
                if (input.equals("quit")) {
                    break;
                }

                String[] dimensions = input.split(" ");
                int m = Integer.parseInt(dimensions[0]);
                int n = Integer.parseInt(dimensions[1]);
                int p = Integer.parseInt(dimensions[2]);

                float[][] A = new float[m][n];
                float[][] B = new float[n][p];
                float[][] C = new float[m][p];

                Random rand = new Random();
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        A[i][j] = rand.nextFloat();
                    }
                }
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < p; j++) {
                        B[i][j] = rand.nextFloat();
                    }
                }

                int repetitions = 10; // 运行次数
                double total=0,timeSpent=0;
                for (int k = 0; k < repetitions; k++) {
                    long start = System.nanoTime();
                    matrixMultiply(A, B, C);
                    long end = System.nanoTime();
                    timeSpent = (end - start) / 1e9;
                    total+=timeSpent;
                    
                }

                System.out.printf("avg time spent for matrix multiplication (C_%dx%d = A_%dx%d * B_%dx%d): %f seconds\n", m, p, m, n, n, p, total/10);
                writer.write(String.format("%dx%d\t%dx%d\t%f\n", m, n, n, p, total/10));
            }

            scanner.close();
            writer.close();
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }

}