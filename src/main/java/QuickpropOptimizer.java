import java.util.HashMap;
import java.util.Map;

public class QuickpropOptimizer implements Optimizer {
    private final double epsilon = 1e-8;
    private final double maxStep = 1.0;

    private final Map<Neuron, double[]> prevGradients = new HashMap<>();
    private final Map<Neuron, double[]> prevWeightUpdates = new HashMap<>();

    private final Map<Neuron, Double> prevGradBias = new HashMap<>();
    private final Map<Neuron, Double> prevBiasUpdate = new HashMap<>();

    @Override
    public void update(Neuron neuron, double[] inputs, double learningRate) {
        int n = inputs.length;

        prevGradients.putIfAbsent(neuron, new double[n]);
        prevWeightUpdates.putIfAbsent(neuron, new double[n]);

        prevGradBias.putIfAbsent(neuron, 0.0);
        prevBiasUpdate.putIfAbsent(neuron, 0.0);

        double delta = neuron.getDelta();

        double[] lastGrad = prevGradients.get(neuron);
        double[] lastDelta = prevWeightUpdates.get(neuron);

        for (int i = 0; i < n; i++) {
            double grad = delta * inputs[i];
            double gradPrev = lastGrad[i];
            double deltaPrev = lastDelta[i];

            double weightUpdate;

            if (Math.abs(grad - gradPrev) < epsilon) {
                // fallback — обычный шаг
                weightUpdate = -learningRate * grad;
            } else {
                weightUpdate = grad / (gradPrev - grad + epsilon) * deltaPrev;
            }

            // Ограничиваем максимальный шаг
            weightUpdate = Math.max(-maxStep, Math.min(maxStep, weightUpdate));

            neuron.setWeight(i, neuron.getWeights()[i] + weightUpdate);

            lastGrad[i] = grad;
            lastDelta[i] = weightUpdate;
        }

        // Bias
        double gradBias = delta;
        double gradBiasPrev = prevGradBias.get(neuron);
        double biasDeltaPrev = prevBiasUpdate.get(neuron);

        double biasUpdate;
        if (Math.abs(gradBias - gradBiasPrev) < epsilon) {
            biasUpdate = -learningRate * gradBias;
        } else {
            biasUpdate = gradBias / (gradBiasPrev - gradBias + epsilon) * biasDeltaPrev;
        }

        biasUpdate = Math.max(-maxStep, Math.min(maxStep, biasUpdate));

        neuron.setBias(neuron.getBias() + biasUpdate);

        prevGradBias.put(neuron, gradBias);
        prevBiasUpdate.put(neuron, biasUpdate);
    }
}
