package utility;

public class MatrixUtils {
    public static double convolveRegion(double[][] input, double[][] kernel, int x, int y) {
        int kernelWidth = kernel.length;
        int kernelHeight = kernel[0].length;
        double sum = 0.0;

        for (int i = 0; i < kernelWidth; i++) {
            for (int j = 0; j < kernelHeight; j++) {
                sum += input[x + i][y + j] * kernel[i][j];
            }
        }

        return sum;
    }

    // Свёртка входного изображения с ядром
    public static double[][] convolve(double[][] input, double[][] kernel, int stride) {
        int inputWidth = input.length;
        int inputHeight = input[0].length;
        int kernelWidth = kernel.length;
        int kernelHeight = kernel[0].length;

        int outputWidth = (inputWidth - kernelWidth) / stride + 1;
        int outputHeight = (inputHeight - kernelHeight) / stride + 1;

        double[][] output = new double[outputWidth][outputHeight];

        for (int i = 0; i < outputWidth; i++) {
            for (int j = 0; j < outputHeight; j++) {
                output[i][j] = convolveRegion(input, kernel, i * stride, j * stride);
            }
        }

        return output;
    }

    // Перегрузка с stride по умолчанию = 1
    public static double[][] convolve(double[][] input, double[][] kernel) {
        return convolve(input, kernel, 1);
    }
}
