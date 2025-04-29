package service;

import lombok.Getter;
import lombok.Setter;
import service.activationfunction.ActivationFunction;

import java.util.Random;

@Getter
@Setter
public class Layer {

    private Neuron[] neurons;

    public Layer(int numberOfNeurons, int inputSizePerNeuron, ActivationFunction activationFunction, Random random) {
        neurons = new Neuron[numberOfNeurons];
        for (int i = 0; i < numberOfNeurons; i++) {
            neurons[i] = new Neuron(inputSizePerNeuron, activationFunction, random);
        }
    }

    public double[] forward(double[] inputs) {
        double[] outputs = new double[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            outputs[i] = neurons[i].forward(inputs);
        }
        return outputs;
    }
}
