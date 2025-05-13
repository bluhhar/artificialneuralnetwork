package service;

public class SoftmaxLayer {

    public double[] forward(double[] input) {
        // Численно стабильная реализация softmax
        double max = Double.NEGATIVE_INFINITY;
        for (double val : input) {
            if (val > max) max = val;
        }

        double[] expValues = new double[input.length];
        double sum = 0.0;

        for (int i = 0; i < input.length; i++) {
            expValues[i] = Math.exp(input[i] - max);
            sum += expValues[i];
        }

        sum = Math.max(sum, 1e-15); // предотвращаем деление на 0

        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = expValues[i] / sum;
        }

        return output;
    }
}
