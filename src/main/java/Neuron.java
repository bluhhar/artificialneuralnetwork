import lombok.Getter;
import lombok.Setter;
import service.activationfunction.ActivationFunction;

import java.util.Random;

@Getter
@Setter
public class Neuron {
    private double[] weights;
    private double bias;
    private ActivationFunction activationFunction;
    private double output;
    private double inputSum; // сумма взвешенных входов
    private double delta;

    public Neuron(int inputSize, ActivationFunction activationFunction, Random random) {
        this.weights = new double[inputSize];
        this.bias = random.nextDouble() - 0.5;
        this.activationFunction = activationFunction;

        for (int i = 0; i < inputSize; i++) {
            weights[i] = random.nextDouble() - 0.5;
        }
    }

    public double forward(double[] inputs) {
        inputSum = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            inputSum += inputs[i] * weights[i];
        }
        inputSum += bias;
        output = activationFunction.activate(inputSum);
        return output;
    }

    public void setWeight(int index, double value) {
        weights[index] = value;
    }
}
