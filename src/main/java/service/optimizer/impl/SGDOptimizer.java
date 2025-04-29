package service.optimizer.impl;

import service.Neuron;
import service.optimizer.Optimizer;
import service.regularizer.Regularizer;

public class SGDOptimizer implements Optimizer {
    private Regularizer regularizer;

    public SGDOptimizer() {
        this.regularizer = null;
    }

    public SGDOptimizer(Regularizer regularizer) {
        this.regularizer = regularizer;
    }

    @Override
    public void update(Neuron neuron, double[] inputs, double learningRate) {
        double delta = neuron.getDelta();
        double[] weights = neuron.getWeights();

        double[] regGradient = null;
        if (regularizer != null) {
            regGradient = regularizer.computeGradient(weights);
        }

        for (int i = 0; i < weights.length; i++) {
            double grad = delta * inputs[i];
            if (regGradient != null) {
                grad += regGradient[i];
            }
            neuron.setWeight(i, weights[i] + learningRate * grad);
        }

        neuron.setBias(neuron.getBias() + learningRate * delta);
    }
}
