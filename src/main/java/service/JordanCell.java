package service;

import lombok.Getter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

@Getter
public class JordanCell {
    private final int inputSize;
    private final int contextSize;
    private final int hiddenSize;
    private final int outputSize;

    private final double[][] weightsInputHidden;
    private final double[][] weightsContextHidden;
    private final double[][] weightsHiddenOutput;

    private final double[] biasHidden;
    private final double[] biasOutput;

    private double[] contextState;
    private final List<double[]> hiddenHistory = new ArrayList<>();
    private final List<double[]> contextHistory = new ArrayList<>();

    private final Random random = new Random();

    public JordanCell(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.contextSize = outputSize;

        this.weightsInputHidden = randomMatrix(hiddenSize, inputSize);
        this.weightsContextHidden = randomMatrix(hiddenSize, contextSize);
        this.weightsHiddenOutput = randomMatrix(outputSize, hiddenSize);

        this.biasHidden = randomArray(hiddenSize);
        this.biasOutput = randomArray(outputSize);

        this.contextState = new double[contextSize];
    }

    public double[] forward(double[] input) {
        double[] hiddenInput = new double[hiddenSize];

        for (int i = 0; i < hiddenSize; i++) {
            hiddenInput[i] = biasHidden[i];
            for (int j = 0; j < inputSize; j++) {
                hiddenInput[i] += weightsInputHidden[i][j] * input[j];
            }
            for (int j = 0; j < contextSize; j++) {
                hiddenInput[i] += weightsContextHidden[i][j] * contextState[j];
            }
            hiddenInput[i] = sigmoid(hiddenInput[i]);
        }

        double[] output = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            output[i] = biasOutput[i];
            for (int j = 0; j < hiddenSize; j++) {
                output[i] += weightsHiddenOutput[i][j] * hiddenInput[j];
            }
            output[i] = sigmoid(output[i]);
        }

        // Сохраняем состояния для BPTT
        hiddenHistory.add(Arrays.copyOf(hiddenInput, hiddenInput.length));
        contextHistory.add(Arrays.copyOf(contextState, contextState.length));

        // Обновляем контекст
        contextState = Arrays.copyOf(output, output.length);

        return output;
    }

    public void backward(List<double[]> inputs, List<double[]> targets, double learningRate) {
        int T = inputs.size();

        // Инициализация градиентов
        double[][] dW_ih = new double[hiddenSize][inputSize];
        double[][] dW_ch = new double[hiddenSize][contextSize];
        double[][] dW_ho = new double[outputSize][hiddenSize];
        double[] dB_h = new double[hiddenSize];
        double[] dB_o = new double[outputSize];

        double[] dContextNext = new double[contextSize];

        for (int t = T - 1; t >= 0; t--) {
            double[] input = inputs.get(t);
            double[] target = targets.get(t);
            double[] hidden = hiddenHistory.get(t);
            double[] context = contextHistory.get(t);

            // --- Выход ---
            double[] output = contextState; // contextState содержит output из последнего шага
            double[] dOut = new double[outputSize];
            for (int i = 0; i < outputSize; i++) {
                double y = output[i];
                double error = y - target[i];
                dOut[i] = error * y * (1 - y); // сигмоида
                dB_o[i] += dOut[i];

                for (int j = 0; j < hiddenSize; j++) {
                    dW_ho[i][j] += dOut[i] * hidden[j];
                }
            }

            // --- Скрытый слой ---
            double[] dHidden = new double[hiddenSize];
            for (int i = 0; i < hiddenSize; i++) {
                double h = hidden[i];
                double sum = 0.0;
                for (int j = 0; j < outputSize; j++) {
                    sum += weightsHiddenOutput[j][i] * dOut[j];
                }
                dHidden[i] = sum * h * (1 - h);
                dB_h[i] += dHidden[i];

                for (int j = 0; j < inputSize; j++) {
                    dW_ih[i][j] += dHidden[i] * input[j];
                }
                for (int j = 0; j < contextSize; j++) {
                    dW_ch[i][j] += dHidden[i] * context[j];
                }
            }

            // Простой перенос ошибки для контекста (опционально)
            dContextNext = Arrays.copyOf(dOut, dOut.length);
        }

        // Обновление весов и bias
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weightsInputHidden[i][j] -= learningRate * dW_ih[i][j];
            }
            for (int j = 0; j < contextSize; j++) {
                weightsContextHidden[i][j] -= learningRate * dW_ch[i][j];
            }
            biasHidden[i] -= learningRate * dB_h[i];
        }

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsHiddenOutput[i][j] -= learningRate * dW_ho[i][j];
            }
            biasOutput[i] -= learningRate * dB_o[i];
        }

        // Очистка истории
        hiddenHistory.clear();
        contextHistory.clear();
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private double[] randomArray(int size) {
        double[] arr = new double[size];
        for (int i = 0; i < size; i++) arr[i] = random.nextDouble() - 0.5;
        return arr;
    }

    private double[][] randomMatrix(int rows, int cols) {
        double[][] mat = new double[rows][cols];
        for (int i = 0; i < rows; i++) mat[i] = randomArray(cols);
        return mat;
    }
}
