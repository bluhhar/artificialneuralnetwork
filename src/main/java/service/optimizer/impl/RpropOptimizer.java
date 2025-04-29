package service.optimizer.impl;

import service.Neuron;
import service.optimizer.Optimizer;

import java.util.HashMap;
import java.util.Map;

public class RpropOptimizer implements Optimizer {
    private final double etaPlus = 1.2;
    private final double etaMinus = 0.5;
    private final double deltaInit = 0.1;
    private final double deltaMin = 1e-6;
    private final double deltaMax = 50.0;

    private final Map<Neuron, double[]> previousGradients = new HashMap<>();
    private final Map<Neuron, double[]> stepSizes = new HashMap<>();

    private final Map<Neuron, Double> prevGradBias = new HashMap<>();
    private final Map<Neuron, Double> stepBias = new HashMap<>();

    @Override
    public void update(Neuron neuron, double[] inputs, double learningRateIgnored) {
        int inputSize = inputs.length;
        double delta = neuron.getDelta();

        // --- Инициализация хранилищ ---
        previousGradients.putIfAbsent(neuron, new double[inputSize]);
        stepSizes.putIfAbsent(neuron, initArray(inputSize, deltaInit));

        prevGradBias.putIfAbsent(neuron, 0.0);
        stepBias.putIfAbsent(neuron, deltaInit);

        double[] prevGrad = previousGradients.get(neuron);
        double[] deltas = stepSizes.get(neuron);

        for (int i = 0; i < inputSize; i++) {
            double grad = delta * inputs[i];
            double sign = grad * prevGrad[i];

            if (sign > 0) {
                deltas[i] = Math.min(deltas[i] * etaPlus, deltaMax);
                neuron.setWeight(i, neuron.getWeights()[i] - Math.signum(grad) * deltas[i]);
                prevGrad[i] = grad;
            } else if (sign < 0) {
                deltas[i] = Math.max(deltas[i] * etaMinus, deltaMin);
                prevGrad[i] = 0.0; // сброс градиента
            } else {
                // знак не изменился или один из градиентов = 0
                neuron.setWeight(i, neuron.getWeights()[i] - Math.signum(grad) * deltas[i]);
                prevGrad[i] = grad;
            }
        }

        // --- Bias ---
        double gradBias = delta;
        double signBias = gradBias * prevGradBias.get(neuron);
        double deltaBias = stepBias.get(neuron);

        if (signBias > 0) {
            deltaBias = Math.min(deltaBias * etaPlus, deltaMax);
            neuron.setBias(neuron.getBias() - Math.signum(gradBias) * deltaBias);
            prevGradBias.put(neuron, gradBias);
        } else if (signBias < 0) {
            deltaBias = Math.max(deltaBias * etaMinus, deltaMin);
            prevGradBias.put(neuron, 0.0);
        } else {
            neuron.setBias(neuron.getBias() - Math.signum(gradBias) * deltaBias);
            prevGradBias.put(neuron, gradBias);
        }

        stepBias.put(neuron, deltaBias);
    }

    private double[] initArray(int size, double value) {
        double[] arr = new double[size];
        for (int i = 0; i < size; i++) arr[i] = value;
        return arr;
    }
}