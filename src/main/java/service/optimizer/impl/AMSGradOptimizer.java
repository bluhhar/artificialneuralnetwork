package service.optimizer.impl;

import lombok.AllArgsConstructor;
import service.Neuron;
import service.optimizer.Optimizer;
import service.regularizer.Regularizer;

import java.util.HashMap;
import java.util.Map;

@AllArgsConstructor
public class AMSGradOptimizer implements Optimizer {

    private Regularizer regularizer;

    private double betaOne;
    private double betaTwo;
    private final double epsilon = 1e-8;

    private final Map<Neuron, double[]> m = new HashMap<>();
    private final Map<Neuron, double[]> v = new HashMap<>();
    private final Map<Neuron, double[]> vHat = new HashMap<>();
    private final Map<Neuron, Double> biasM = new HashMap<>();
    private final Map<Neuron, Double> biasV = new HashMap<>();
    private final Map<Neuron, Double> biasVHat = new HashMap<>();

    public AMSGradOptimizer(double betaOne, double betaTwo) {
        this.betaOne = betaOne;
        this.betaTwo = betaTwo;
    }

    @Override
    public void update(Neuron neuron, double[] inputs, double learningRate) {
        int n = inputs.length;

        m.putIfAbsent(neuron, new double[n]);
        v.putIfAbsent(neuron, new double[n]);
        vHat.putIfAbsent(neuron, new double[n]);
        biasM.putIfAbsent(neuron, 0.0);
        biasV.putIfAbsent(neuron, 0.0);
        biasVHat.putIfAbsent(neuron, 0.0);

        double[] mNeuron = m.get(neuron);
        double[] vNeuron = v.get(neuron);
        double[] vHatNeuron = vHat.get(neuron);

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

            mNeuron[i] = betaOne * mNeuron[i] + (1 - betaOne) * grad;
            vNeuron[i] = betaTwo * vNeuron[i] + (1 - betaTwo) * grad * grad;
            vHatNeuron[i] = Math.max(vHatNeuron[i], vNeuron[i]);

            double correctedM = mNeuron[i]; // без bias correction для простоты
            double correctedV = vHatNeuron[i];

            double weightUpdate = learningRate * correctedM / (Math.sqrt(correctedV) + epsilon);
            neuron.setWeight(i, neuron.getWeights()[i] + weightUpdate);
        }

        // Отдельно обновляем bias
        double gradBias = delta;

        double mBias = betaOne * biasM.get(neuron) + (1 - betaOne) * gradBias;
        double vBias = betaTwo * biasV.get(neuron) + (1 - betaTwo) * gradBias * gradBias;
        double vHatBias = Math.max(biasVHat.get(neuron), vBias);

        biasM.put(neuron, mBias);
        biasV.put(neuron, vBias);
        biasVHat.put(neuron, vHatBias);

        double correctedMBias = mBias;
        double correctedVBias = vHatBias;

        double biasUpdate = learningRate * correctedMBias / (Math.sqrt(correctedVBias) + epsilon);
        neuron.setBias(neuron.getBias() + biasUpdate);
    }
}