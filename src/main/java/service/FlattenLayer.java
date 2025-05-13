package service;

public class FlattenLayer {

    private int depth;
    private int width;
    private int height;

    // Прямой проход: свёртка 3D-массива в 1D
    public double[] forward(double[][][] input) {
        depth = input.length;
        width = input[0].length;
        height = input[0][0].length;

        double[] output = new double[depth * width * height];
        int index = 0;

        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < width; i++) {
                for (int j = 0; j < height; j++) {
                    output[index++] = input[d][i][j];
                }
            }
        }

        return output;
    }

    // Обратный проход: разворачиваем 1D обратно в 3D
    public double[][][] backward(double[] dOut) {
        double[][][] output = new double[depth][width][height];
        int index = 0;

        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < width; i++) {
                for (int j = 0; j < height; j++) {
                    output[d][i][j] = dOut[index++];
                }
            }
        }

        return output;
    }
}
