public class SGDOptimizer implements Optimizer {
    @Override
    public void update(Neuron neuron, double[] inputs, double learningRate) {
        double delta = neuron.getDelta();
        for (int i = 0; i < neuron.getWeights().length; i++) {
            double oldWeight = neuron.getWeights()[i];
            neuron.setWeight(i, oldWeight + learningRate * delta * inputs[i]);
        }
        neuron.setBias(neuron.getBias() + learningRate * delta);
    }
}