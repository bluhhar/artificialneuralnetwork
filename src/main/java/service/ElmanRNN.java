package service;

import java.util.Random;

public class ElmanRNN {
    int inputSize;
    int hiddenSize;
    int outputSize;

    float[][] wInputHidden;   // [hidden][input]
    float[][] wContextHidden; // [hidden][hidden]
    float[][] wHiddenOutput;  // [output][hidden]

    float[] context; // предыдущие скрытые состояния

    Random random = new Random();

    public ElmanRNN(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        wInputHidden = new float[hiddenSize][inputSize];
        wContextHidden = new float[hiddenSize][hiddenSize];
        wHiddenOutput = new float[outputSize][hiddenSize];

        context = new float[hiddenSize];

        initWeights(wInputHidden);
        initWeights(wContextHidden);
        initWeights(wHiddenOutput);
    }

    private void initWeights(float[][] weights) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = (random.nextFloat() - 0.5f) * 2f; // [-1, 1]
            }
        }
    }

    public float[] forward(float[] input) {
        float[] hidden = new float[hiddenSize];

        // Вход + контекст
        for (int i = 0; i < hiddenSize; i++) {
            float sum = 0.0f;
            for (int j = 0; j < inputSize; j++) {
                sum += input[j] * wInputHidden[i][j];
            }
            for (int j = 0; j < hiddenSize; j++) {
                sum += context[j] * wContextHidden[i][j];
            }
            hidden[i] = (float) Math.tanh(sum);
        }

        // Копируем hidden в context на следующий шаг
        System.arraycopy(hidden, 0, context, 0, hiddenSize);

        float[] output = new float[outputSize];
        for (int i = 0; i < outputSize; i++) {
            float sum = 0.0f;
            for (int j = 0; j < hiddenSize; j++) {
                sum += hidden[j] * wHiddenOutput[i][j];
            }
            output[i] = sigmoid(sum); // бинарный выход
        }

        return output;
    }

    public void train(float[][] inputs, float[][] targets, int epochs, float learningRate) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            float totalLoss = 0.0f;

            for (int t = 0; t < inputs.length; t++) {
                float[] input = inputs[t];
                float[] target = targets[t];

                // --- FORWARD ---
                float[] hidden = new float[hiddenSize];
                for (int i = 0; i < hiddenSize; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < inputSize; j++) {
                        sum += input[j] * wInputHidden[i][j];
                    }
                    for (int j = 0; j < hiddenSize; j++) {
                        sum += context[j] * wContextHidden[i][j];
                    }
                    hidden[i] = (float) Math.tanh(sum);
                }

                // Копируем hidden в context на следующий шаг
                System.arraycopy(hidden, 0, context, 0, hiddenSize);

                float[] output = new float[outputSize];
                for (int i = 0; i < outputSize; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < hiddenSize; j++) {
                        sum += hidden[j] * wHiddenOutput[i][j];
                    }
                    output[i] = sigmoid(sum);
                }

                // --- BACKWARD ---
                float[] outputErrors = new float[outputSize];
                for (int i = 0; i < outputSize; i++) {
                    outputErrors[i] = (target[i] - output[i]) * output[i] * (1 - output[i]); // dE/dy
                    totalLoss += Math.pow(target[i] - output[i], 2);
                }

                float[] hiddenErrors = new float[hiddenSize];
                for (int i = 0; i < hiddenSize; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < outputSize; j++) {
                        sum += outputErrors[j] * wHiddenOutput[j][i];
                    }
                    hiddenErrors[i] = (1 - hidden[i] * hidden[i]) * sum; // tanh' = 1 - x²
                }

                // --- UPDATE WEIGHTS ---
                for (int i = 0; i < outputSize; i++) {
                    for (int j = 0; j < hiddenSize; j++) {
                        wHiddenOutput[i][j] += learningRate * outputErrors[i] * hidden[j];
                    }
                }

                for (int i = 0; i < hiddenSize; i++) {
                    for (int j = 0; j < inputSize; j++) {
                        wInputHidden[i][j] += learningRate * hiddenErrors[i] * input[j];
                    }
                    for (int j = 0; j < hiddenSize; j++) {
                        wContextHidden[i][j] += learningRate * hiddenErrors[i] * context[j];
                    }
                }
            }

            if (epoch % 10 == 0) {
                System.out.println("Epoch " + epoch + ", Loss: " + totalLoss / inputs.length);
            }
        }
    }

    private float sigmoid(float x) {
        return (float)(1.0 / (1.0 + Math.exp(-x)));
    }

    public void resetContext() {
        for (int i = 0; i < context.length; i++) {
            context[i] = 0.0f;
        }
    }

    public void evaluate(float[][] inputs, float[][] targets) {
        int correct = 0;

        for (int i = 0; i < inputs.length; i++) {
            float[] input = inputs[i];
            float[] target = targets[i];

            float[] output = forward(input);

            int predicted = output[0] >= 0.5f ? 1 : 0;
            int actual = target[0] >= 0.5f ? 1 : 0;

            if (predicted == actual) correct++;

            System.out.printf("Sample %d: Target = %d, Predicted = %.2f (%d)%n",
                    i, actual, output[0], predicted);
        }

        float accuracy = (correct * 100.0f) / inputs.length;
        System.out.printf("Accuracy: %.2f%% (%d/%d)%n", accuracy, correct, inputs.length);
    }
}
