package service.optimizer.impl;

import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import service.Neuron;
import service.optimizer.Optimizer;
import service.regularizer.Regularizer;

import java.util.HashMap;
import java.util.Map;

@AllArgsConstructor
@NoArgsConstructor
public class RmspropGravesOptimizer implements Optimizer {

    private Regularizer regularizer;

    private final double rho = 0.95;
    private final double epsilon = 1e-4;

    private final Map<Neuron, double[]> eg = new HashMap<>();
    private final Map<Neuron, double[]> eg2 = new HashMap<>();

    private final Map<Neuron, Double> egBias = new HashMap<>();
    private final Map<Neuron, Double> eg2Bias = new HashMap<>();

    @Override
    public void update(Neuron neuron, double[] inputs, double learningRate) {
        int n = inputs.length;

        eg.putIfAbsent(neuron, new double[n]);
        eg2.putIfAbsent(neuron, new double[n]);
        egBias.putIfAbsent(neuron, 0.0);
        eg2Bias.putIfAbsent(neuron, 0.0);

        double[] egNeuron = eg.get(neuron);
        double[] eg2Neuron = eg2.get(neuron);

        double delta = neuron.getDelta();

        double[] weights = neuron.getWeights();

        double[] regGradient = null;
        if (regularizer != null) {
            regGradient = regularizer.computeGradient(weights);
        }

        for (int i = 0; i < n; i++) {
            double grad = delta * inputs[i];
            if (regGradient != null) {
                grad += regGradient[i];
            }

            egNeuron[i] = rho * egNeuron[i] + (1 - rho) * grad;
            eg2Neuron[i] = rho * eg2Neuron[i] + (1 - rho) * grad * grad;

            double denom = Math.sqrt(eg2Neuron[i] - egNeuron[i] * egNeuron[i] + epsilon);
            double update = learningRate * grad / denom;

            neuron.setWeight(i, neuron.getWeights()[i] + update);
        }

        // Bias
        double gradBias = delta;

        double prevEg = egBias.get(neuron);
        double prevEg2 = eg2Bias.get(neuron);

        double newEg = rho * prevEg + (1 - rho) * gradBias;
        double newEg2 = rho * prevEg2 + (1 - rho) * gradBias * gradBias;

        double denomBias = Math.sqrt(newEg2 - newEg * newEg + epsilon);
        double updateBias = learningRate * gradBias / denomBias;

        neuron.setBias(neuron.getBias() + updateBias);

        egBias.put(neuron, newEg);
        eg2Bias.put(neuron, newEg2);
    }
}
