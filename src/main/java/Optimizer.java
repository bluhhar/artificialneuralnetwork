public interface Optimizer {

    void update(Neuron neuron, double[] inputs, double learningRate);

}
