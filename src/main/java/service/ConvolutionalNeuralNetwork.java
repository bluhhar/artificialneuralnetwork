package service;

import utility.KnowledgeDistiller;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ConvolutionalNeuralNetwork {
    private final ConvLayer convLayer;
    private final PoolingLayer poolingLayer;
    private final FlattenLayer flattenLayer;
    private final FullyConnectedLayer fcLayer;
    private final SoftmaxLayer softmaxLayer;

    public ConvolutionalNeuralNetwork(int inputWidth, int inputHeight, int kernelSize,
                                      int numKernels, int fcOutputSize, double learningRate) {
        this.convLayer = new ConvLayer(kernelSize, numKernels, learningRate, 1);
        this.poolingLayer = new PoolingLayer();

        int convOutputWidth = inputWidth - kernelSize + 1;
        int convOutputHeight = inputHeight - kernelSize + 1;
        int pooledWidth = convOutputWidth / 2;
        int pooledHeight = convOutputHeight / 2;
        int flattenedSize = numKernels * pooledWidth * pooledHeight;

        this.flattenLayer = new FlattenLayer();
        this.fcLayer = new FullyConnectedLayer(flattenedSize, fcOutputSize, learningRate);
        this.softmaxLayer = new SoftmaxLayer();
    }

    public void train(double[][] input, double[] target) {
        List<double[][]> convOutput = convLayer.forward(input);
        double[][][] convOutput3D = convertTo3DArray(convOutput);
        double[][][] pooledOutput = poolingLayer.forward(convOutput3D);
        double[] flattened = flattenLayer.forward(pooledOutput);
        double[] fcOutput = fcLayer.forward(flattened);
        double[] output = softmaxLayer.forward(fcOutput);

        // Проверка числовой стабильности
        for (double val : output) {
            if (Double.isNaN(val) || Double.isInfinite(val)) {
                throw new RuntimeException("Numerical instability in forward pass");
            }
        }

        // Кросс-энтропия + softmax градиент
        double[] lossGradient = new double[output.length];
        for (int i = 0; i < output.length; i++) {
            lossGradient[i] = (output[i] - target[i]) / output.length;
            lossGradient[i] = Math.max(-1.0, Math.min(1.0, lossGradient[i])); // gradient clipping
        }

        try {
            double[] fcGrad = fcLayer.backward(lossGradient);
            double[][][] flattenGrad = flattenLayer.backward(fcGrad);
            double[][][] poolGrad = poolingLayer.backward(flattenGrad);
            convLayer.backward(convertTo2DList(poolGrad));
        } catch (Exception ex) {
            throw new RuntimeException("Backpropagation failed: " + ex.getMessage());
        }

        checkWeights();
    }

    private void checkWeights() {
        for (double[][] kernel : convLayer.getKernels()) {
            for (double[] row : kernel) {
                for (double w : row) {
                    if (Double.isNaN(w) || Double.isInfinite(w)) {
                        throw new RuntimeException("Invalid values in conv kernels");
                    }
                }
            }
        }

        for (double[] row : fcLayer.getWeights()) {
            for (double w : row) {
                if (Double.isNaN(w) || Double.isInfinite(w)) {
                    throw new RuntimeException("Invalid values in FC weights");
                }
            }
        }
    }

    public double[] forward(double[][] input) {
        List<double[][]> convOutput = convLayer.forward(input);
        double[][][] convOutput3D = convertTo3DArray(convOutput);
        double[][][] pooledOutput = poolingLayer.forward(convOutput3D);
        double[] flattened = flattenLayer.forward(pooledOutput);
        double[] fcOutput = fcLayer.forward(flattened);
        return softmaxLayer.forward(fcOutput);
    }

    public double calculateLoss(double[] predicted, double[] target) {
        double loss = 0;
        for (int i = 0; i < predicted.length; i++) {
            loss -= target[i] * Math.log(Math.max(predicted[i], 1e-15));
        }
        return loss;
    }

    public double evaluate(List<Pair<double[][], double[]>> testSet) {
        int correct = 0;
        for (Pair<double[][], double[]> sample : testSet) {
            double[] output = forward(sample.first());
            int predicted = argMax(output);
            int actual = argMax(sample.second());
            if (predicted == actual) correct++;
        }
        return (double) correct / testSet.size();
    }

    private int argMax(double[] array) {
        int idx = 0;
        double max = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
                idx = i;
            }
        }
        return idx;
    }

    public double[][][] convertTo3DArray(List<double[][]> convOutput) {
        int depth = convOutput.size();
        int width = convOutput.get(0).length;
        int height = convOutput.get(0)[0].length;

        double[][][] result = new double[depth][width][height];
        for (int d = 0; d < depth; d++) {
            double[][] matrix = convOutput.get(d);
            for (int i = 0; i < width; i++) {
                System.arraycopy(matrix[i], 0, result[d][i], 0, height);
            }
        }

        return result;
    }

    public List<double[][]> convertTo2DList(double[][][] input) {
        List<double[][]> result = new ArrayList<>();
        for (double[][] matrix : input) {
            result.add(matrix);
        }
        return result;
    }
}
