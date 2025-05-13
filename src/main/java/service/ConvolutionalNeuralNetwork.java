package service;

import utility.KnowledgeDistiller;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ConvolutionalNeuralNetwork {
    public ConvLayer convLayer;
    private PoolingLayer poolingLayer;
    private FlattenLayer flattenLayer;
    public FullyConnectedLayer fcLayer;
    private SoftmaxLayer softmaxLayer;
    public int kernelSize;
    public KnowledgeDistiller distiller = new KnowledgeDistiller();

    public boolean enablePruning;
    public boolean enableQuantization;
    public boolean enableDistillation;

    public double pruningSparsity = 0.2;
    public int pruningFrequency = 5;
    public double distillationTemperature = 2.0;

    public ConvolutionalNeuralNetwork(int inputWidth, int inputHeight, int kernelSize,
                                      int numKernels, int fcOutputSize, double learningRate,
                                      boolean enablePruning, boolean enableQuantization, boolean enableDistillation) {
        this.enablePruning = enablePruning;
        this.enableQuantization = enableQuantization;
        this.enableDistillation = enableDistillation;
        this.kernelSize = kernelSize;

        convLayer = new ConvLayer(kernelSize, numKernels, learningRate, 1);
        poolingLayer = new PoolingLayer(2, 2);

        int convOutputWidth = inputWidth - kernelSize + 1;
        int convOutputHeight = inputHeight - kernelSize + 1;
        int pooledWidth = convOutputWidth / 2;
        int pooledHeight = convOutputHeight / 2;

        int flattenedSize = numKernels * pooledWidth * pooledHeight;

        flattenLayer = new FlattenLayer();
        fcLayer = new FullyConnectedLayer(flattenedSize, fcOutputSize, learningRate);
        softmaxLayer = new SoftmaxLayer();
    }

    public void train(double[][] input, double[] target, ConvolutionalNeuralNetwork teacherModel) {
        List<double[][]> convOutput = convLayer.forward(input);
        double[][][] convOutput3D = convertTo3DArray(convOutput);
        double[][][] pooledOutput = poolingLayer.forward(convOutput3D);
        double[] flattened = flattenLayer.forward(pooledOutput);
        double[] fcOutput = fcLayer.forward(flattened);
        double[] studentOutput = softmaxLayer.forward(fcOutput);

        if (Arrays.stream(studentOutput).anyMatch(x -> Double.isNaN(x) || Double.isInfinite(x))) {
            throw new RuntimeException("Numerical instability in forward pass");
        }

        double[] lossGradient;
        if (enableDistillation && teacherModel != null) {
            double[] teacherOutput = teacherModel.forward(input);
            lossGradient = distiller.calculateGradient(studentOutput, teacherOutput, target);
        } else {
            lossGradient = new double[studentOutput.length];
            for (int i = 0; i < studentOutput.length; i++) {
                lossGradient[i] = (studentOutput[i] - target[i]) / studentOutput.length;
            }
        }

        for (int i = 0; i < lossGradient.length; i++) {
            lossGradient[i] = Math.max(-1.0, Math.min(1.0, lossGradient[i]));
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

    public double[][][] convertTo3DArray(List<double[][]> convOutput) {
        int depth = convOutput.size();
        int width = convOutput.getFirst().length;
        int height = convOutput.getFirst()[0].length;

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
        return new ArrayList<>(Arrays.asList(input));
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
            if (predicted == actual) {
                correct++;
            }
        }
        return (double) correct / testSet.size();
    }

    private int argMax(double[] array) {
        int maxIndex = 0;
        double max = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public void prune(double sparsity) {
        if (enablePruning) {
            System.out.println("\n=== Начало прунинга ===");
            convLayer.pruneKernels(sparsity);
            fcLayer.pruneWeights(sparsity);

            int convZero = convLayer.getKernels().stream()
                    .mapToInt(k -> Arrays.stream(k).mapToInt(row -> (int) Arrays.stream(row).filter(w -> w == 0).count()).sum())
                    .sum();
            int fcZero = Arrays.stream(fcLayer.getWeights())
                    .mapToInt(row -> (int) Arrays.stream(row).filter(w -> w == 0).count()).sum();

            System.out.printf("Итог: обнулено %d весов\nConv: %d | FC: %d\n", convZero + fcZero, convZero, fcZero);
        }
    }

    public void quantizeModel() {
        if (enableQuantization) {
            System.out.println("\n=== Начало квантизации ===");
            convLayer.quantizeKernels();
            fcLayer.quantizeWeights();
            System.out.println("=== Квантизация завершена ===");
        }
    }
}

