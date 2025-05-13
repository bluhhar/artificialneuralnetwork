package service;

import lombok.Getter;
import utility.ActivationFunctions;

import java.util.Random;

@Getter
public class FullyConnectedLayer {
    private final int inputSize;
    private final int outputSize;
    private final double[][] weights;
    private final double[] biases;
    private final double learningRate;

    private double[] lastInput;
    private double[] lastPreActivation;
    private double[] lastOutput;

    private final Random rand = new Random();

    public FullyConnectedLayer(int inputSize, int outputSize, double learningRate) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.learningRate = learningRate;

        this.weights = new double[outputSize][inputSize];
        this.biases = new double[outputSize];

        // Xavier/He initialization
        double scale = Math.sqrt(2.0 / (inputSize + outputSize));
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] = (rand.nextDouble() * 2 - 1) * scale;
            }
            biases[i] = 0;
        }
    }

    public double[] forward(double[] input) {
        if (input.length != inputSize) {
            throw new IllegalArgumentException("Input size mismatch.");
        }

        this.lastInput = input;
        this.lastPreActivation = new double[outputSize];
        this.lastOutput = new double[outputSize];

        for (int i = 0; i < outputSize; i++) {
            double sum = biases[i];
            for (int j = 0; j < inputSize; j++) {
                sum += weights[i][j] * input[j];
            }
            lastPreActivation[i] = sum;
            lastOutput[i] = ActivationFunctions.leakyReLU(sum);
        }

        return lastOutput;
    }

    public double[] backward(double[] gradient) {
        double[] inputGradient = new double[inputSize];
        double[] activationGradient = new double[outputSize];

        // dL/dz (где z — до активации)
        for (int i = 0; i < outputSize; i++) {
            activationGradient[i] = gradient[i] * (lastPreActivation[i] > 0 ? 1 : 0);
        }

        // dL/dW и dL/db
        for (int i = 0; i < outputSize; i++) {
            biases[i] -= learningRate * activationGradient[i];
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] -= learningRate * activationGradient[i] * lastInput[j];
            }
        }

        // dL/dInput
        for (int j = 0; j < inputSize; j++) {
            for (int i = 0; i < outputSize; i++) {
                inputGradient[j] += weights[i][j] * activationGradient[i];
            }
        }

        return inputGradient;
    }
}
