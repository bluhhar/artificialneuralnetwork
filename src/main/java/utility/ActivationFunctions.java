package utility;

public class ActivationFunctions {
    // ReLU функция активации с минимальным значением epsilon
    public static double[][] relu(double[][] matrix, double epsilon) {
        int width = matrix.length;
        int height = matrix[0].length;
        double[][] result = new double[width][height];

        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                result[i][j] = Math.max(epsilon, matrix[i][j]);
            }
        }

        return result;
    }

    // Перегрузка с epsilon по умолчанию
    public static double[][] relu(double[][] matrix) {
        return relu(matrix, 1e-7);
    }

    // Leaky ReLU функция
    public static double leakyReLU(double x, double alpha) {
        return x > 0 ? x : alpha * x;
    }

    // Перегрузка с alpha по умолчанию
    public static double leakyReLU(double x) {
        return leakyReLU(x, 0.01);
    }
}
