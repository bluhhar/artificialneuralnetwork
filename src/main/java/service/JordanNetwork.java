package service;

import entity.Iris;

import java.util.ArrayList;
import java.util.List;

public class JordanNetwork {
    private final JordanCell cell;
    private final List<double[]> inputSequence = new ArrayList<>();
    private final List<double[]> targetSequence = new ArrayList<>();
    private final int sequenceLength;

    public JordanNetwork(int inputSize, int hiddenSize, int outputSize, int sequenceLength) {
        this.cell = new JordanCell(inputSize, hiddenSize, outputSize);
        this.sequenceLength = sequenceLength;
    }

    public double[] predict(double[][] sequence) {
        double[] output = null;
        for (double[] stepInput : sequence) {
            output = cell.forward(stepInput);
        }
        return output;
    }

    public void train(Iris sample, double learningRate) {
        // 1. Преобразуем признаки в временную последовательность
        double[][] sequence = toSequence(sample.features);
        inputSequence.clear();
        targetSequence.clear();

        // 2. Forward pass
        for (double[] stepInput : sequence) {
            double[] output = cell.forward(stepInput);
            inputSequence.add(stepInput);
            targetSequence.add(sample.label); // используем один и тот же таргет
        }

        // 3. Backward pass
        cell.backward(inputSequence, targetSequence, learningRate);
    }

    private double[][] toSequence(double[] features) {
        // Разбиваем признаки на временные шаги по одному признаку
        double[][] seq = new double[features.length][1];
        for (int i = 0; i < features.length; i++) {
            seq[i][0] = features[i];
        }
        return seq;
    }
}
