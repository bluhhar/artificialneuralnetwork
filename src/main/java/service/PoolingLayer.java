package service;

public class PoolingLayer {
    private int filterSize;
    private int stride;

    // Координаты максимумов для backpropagation
    private int[][][][] maxIndices;

    public PoolingLayer(int filterSize, int stride) {
        this.filterSize = filterSize;
        this.stride = stride;
    }

    public PoolingLayer() {
        this(2, 2);
    }

    public double[][][] forward(double[][][] input) {
        int depth = input.length;
        int inputWidth = input[0].length;
        int inputHeight = input[0][0].length;

        int outputWidth = (inputWidth - filterSize) / stride + 1;
        int outputHeight = (inputHeight - filterSize) / stride + 1;

        double[][][] output = new double[depth][outputWidth][outputHeight];
        maxIndices = new int[depth][outputWidth][outputHeight][2];

        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < outputWidth; i++) {
                for (int j = 0; j < outputHeight; j++) {
                    double max = Double.NEGATIVE_INFINITY;
                    int maxX = 0, maxY = 0;

                    for (int fi = 0; fi < filterSize; fi++) {
                        for (int fj = 0; fj < filterSize; fj++) {
                            int x = i * stride + fi;
                            int y = j * stride + fj;

                            double val = input[d][x][y];
                            if (val > max) {
                                max = val;
                                maxX = x;
                                maxY = y;
                            }
                        }
                    }

                    output[d][i][j] = max;
                    maxIndices[d][i][j][0] = maxX;
                    maxIndices[d][i][j][1] = maxY;
                }
            }
        }

        return output;
    }

    public double[][][] backward(double[][][] dOut) {
        int depth = dOut.length;
        int inputWidth = maxIndices[0].length * stride;
        int inputHeight = maxIndices[0][0].length * stride;

        double[][][] dInput = new double[depth][inputWidth][inputHeight];

        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < dOut[d].length; i++) {
                for (int j = 0; j < dOut[d][i].length; j++) {
                    int x = maxIndices[d][i][j][0];
                    int y = maxIndices[d][i][j][1];
                    dInput[d][x][y] = dOut[d][i][j];
                }
            }
        }

        return dInput;
    }
}
