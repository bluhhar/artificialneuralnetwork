package service.optimizer.impl;

import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import service.Neuron;
import service.optimizer.Optimizer;
import service.regularizer.Regularizer;

@AllArgsConstructor
@NoArgsConstructor
@Slf4j
public class SGDOptimizer implements Optimizer {
    private Regularizer regularizer;

    @Override
    public void update(Neuron neuron, double[] inputs, double learningRate) {
        //log.info("SGDOptimizer start  update: neuron delta {}, inputs size {}, learningRate {}",
        //        neuron.getDelta(), inputs.length, learningRate);

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
