package service;

import lombok.Getter;
import utility.ActivationFunctions;

import java.util.*;
import java.util.stream.Collectors;

@Getter
public class FullyConnectedLayer {

    private final int inputSize;
    private final int outputSize;
    private final double[][] weights;
    private final double[] biases;
    private final Random rand = new Random();
    private final double learningRate;

    private double[] lastInput;
    private double[] lastPreActivation;
    private double[] lastOutput;

    public FullyConnectedLayer(int inputSize, int outputSize, double learningRate) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.learningRate = learningRate;

        this.weights = new double[outputSize][inputSize];
        this.biases = new double[outputSize];

        // Xavier/Glorot initialization
        double scale = Math.sqrt(2.0 / (inputSize + outputSize));
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] = rand.nextDouble() * scale * 2 - scale;
            }
            biases[i] = 0;
        }
    }

    public double[] forward(double[] input) {
        if (input.length != inputSize) {
            throw new IllegalArgumentException("Input size mismatch");
        }

        lastInput = input;
        lastPreActivation = new double[outputSize];
        double[] output = new double[outputSize];

        for (int i = 0; i < outputSize; i++) {
            double sum = biases[i];
            for (int j = 0; j < inputSize; j++) {
                sum += weights[i][j] * input[j];
            }
            lastPreActivation[i] = sum;
            output[i] = ActivationFunctions.leakyReLU(sum);
        }

        lastOutput = output;
        return output;
    }

    public double[] backward(double[] gradient) {
        double[] inputGradient = new double[inputSize];
        double[][] weightGradients = new double[outputSize][inputSize];
        double[] biasGradients = new double[outputSize];

        double[] activationGradient = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            activationGradient[i] = gradient[i] * (lastPreActivation[i] > 0 ? 1 : 0);
        }

        for (int i = 0; i < outputSize; i++) {
            biasGradients[i] = activationGradient[i];
            for (int j = 0; j < inputSize; j++) {
                weightGradients[i][j] = activationGradient[i] * lastInput[j];
            }
        }

        for (int j = 0; j < inputSize; j++) {
            for (int i = 0; i < outputSize; i++) {
                inputGradient[j] += weights[i][j] * activationGradient[i];
            }
        }

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                if (weights[i][j] != 0) {
                    weights[i][j] -= learningRate * weightGradients[i][j];
                }
            }
        }

        return inputGradient;
    }

    public void pruneWeights(double sparsity) {
        if (sparsity <= 0 || sparsity >= 1) {
            throw new IllegalArgumentException("Sparsity must be in (0, 1)");
        }

        List<Double> absWeights = Arrays.stream(weights)
                .flatMapToDouble(Arrays::stream)
                .filter(w -> w != 0)
                .map(Math::abs)
                .boxed()
                .collect(Collectors.toList());

        if (absWeights.isEmpty() || absWeights.stream().distinct().count() == 1) {
            System.out.println("Предупреждение: Все веса одинаковы, прунинг пропущен.");
            return;
        }

        absWeights.sort(Double::compareTo);
        int thresholdIndex = (int) (sparsity * (absWeights.size() - 1));
        double threshold = absWeights.get(thresholdIndex);

        int nonZeroBefore = (int) Arrays.stream(weights).flatMapToDouble(Arrays::stream).filter(w -> w != 0).count();

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                if (Math.abs(weights[i][j]) < threshold) {
                    weights[i][j] = 0;
                }
            }
        }

        int nonZeroAfter = (int) Arrays.stream(weights).flatMapToDouble(Arrays::stream).filter(w -> w != 0).count();
        System.out.printf("Прунинг полносвязного слоя: было %d, стало %d, удалено %d весов.\n",
                nonZeroBefore, nonZeroAfter, nonZeroBefore - nonZeroAfter);
    }

    public void quantizeWeights() {
        double[] flatWeights = Arrays.stream(weights).flatMapToDouble(Arrays::stream).toArray();
        double min = Arrays.stream(flatWeights).min().orElse(0);
        double max = Arrays.stream(flatWeights).max().orElse(1);
        double scale = (max - min) / 255.0;

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                if (weights[i][j] != 0) {
                    int quantized = (int) ((weights[i][j] - min) / scale);
                    weights[i][j] = min + quantized * scale;
                }
            }
        }

        System.out.printf("Квантизация FC: [%.6f, %.6f] -> 256 уровней\n", min, max);
    }
}

