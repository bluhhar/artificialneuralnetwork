import activationfunction.ActivationFunction;

public class Layer {

    Neuron[] neurons;

    public Layer(int neuronCount, int inputSize, ActivationFunction activation) {
        neurons = new Neuron[neuronCount];
        for (int i = 0; i < neuronCount; i++) {
            neurons[i] = new Neuron(inputSize, activation);
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
