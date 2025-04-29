package service.optimizer;

import service.Neuron;

public interface Optimizer {

    void update(Neuron neuron, double[] inputs, double learningRate);

}
