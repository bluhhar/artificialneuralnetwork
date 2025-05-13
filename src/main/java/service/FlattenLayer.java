package service;

public class FlattenLayer {
    private int depth, width, height;

    // Прямой проход: преобразование 3D -> 1D
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

    // Обратный проход: преобразование 1D -> 3D
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

