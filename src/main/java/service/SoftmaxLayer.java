package service;

import java.util.Arrays;

public class SoftmaxLayer {

    public double[] forward(double[] input) {
        // Численно стабильный softmax
        double max = Arrays.stream(input).max().orElse(0.0);
        double[] expValues = new double[input.length];
        double sum = 0.0;

        for (int i = 0; i < input.length; i++) {
            expValues[i] = Math.exp(input[i] - max);
            sum += expValues[i];
        }

        // Предотвращение деления на ноль
        sum = Math.max(sum, 1e-15);

        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = expValues[i] / sum;
        }

        return output;
    }
}
