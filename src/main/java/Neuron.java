import activationfunction.ActivationFunction;

import java.util.Random;

public class Neuron {
    double[] weights;
    double bias;
    double output;
    double delta;
    ActivationFunction activation;

    public Neuron(int inputSize, ActivationFunction activation) {
        this.weights = new double[inputSize];
        this.bias = Math.random() - 0.5;
        this.activation = activation;
        Random rand = new Random();
        for (int i = 0; i < weights.length; i++) {
            weights[i] = rand.nextGaussian() * 0.01;
        }
    }

    public double forward(double[] inputs) {
        double sum = bias;
        for (int i = 0; i < inputs.length; i++) {
            sum += weights[i] * inputs[i];
        }
        output = activation.activate(sum);
        return output;
    }
}
