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
public class AMSGradOptimizer implements Optimizer {

    private Regularizer regularizer;

    private final double beta1 = 0.9;
    private final double beta2 = 0.999;
    private final double epsilon = 1e-8;

    private final Map<Neuron, double[]> m = new HashMap<>();
    private final Map<Neuron, double[]> v = new HashMap<>();
    private final Map<Neuron, double[]> vHat = new HashMap<>();
    private final Map<Neuron, Double> biasM = new HashMap<>();
    private final Map<Neuron, Double> biasV = new HashMap<>();
    private final Map<Neuron, Double> biasVHat = new HashMap<>();

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

            mNeuron[i] = beta1 * mNeuron[i] + (1 - beta1) * grad;
            vNeuron[i] = beta2 * vNeuron[i] + (1 - beta2) * grad * grad;
            vHatNeuron[i] = Math.max(vHatNeuron[i], vNeuron[i]);

            double correctedM = mNeuron[i]; // без bias correction для простоты
            double correctedV = vHatNeuron[i];

            double weightUpdate = learningRate * correctedM / (Math.sqrt(correctedV) + epsilon);
            neuron.setWeight(i, neuron.getWeights()[i] + weightUpdate);
        }

        // Отдельно обновляем bias
        double gradBias = delta;

        double mBias = beta1 * biasM.get(neuron) + (1 - beta1) * gradBias;
        double vBias = beta2 * biasV.get(neuron) + (1 - beta2) * gradBias * gradBias;
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