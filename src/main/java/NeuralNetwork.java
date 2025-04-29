import lombok.Getter;
import lombok.Setter;
import service.activationfunction.ActivationFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

@Getter
@Setter
public class NeuralNetwork {

    private List<Layer> layers;
    private Random random = new Random();

    public NeuralNetwork() {
        layers = new ArrayList<>();
    }

    public void addLayer(int numberOfNeurons, ActivationFunction activationFunction) {
        int inputSize = layers.isEmpty() ? 4 : layers.get(layers.size() - 1).getNeurons().length;
        layers.add(new Layer(numberOfNeurons, inputSize, activationFunction, random));
    }

    public double[] predict(double[] input) {
        double[] output = input;
        for (Layer layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }

    public void train(double[] input, double[] target, double learningRate) {
        // 1. Прямой проход
        predict(input);

        // 2. Вычисление ошибки на выходном слое
        Layer outputLayer = layers.getLast();
        for (int i = 0; i < outputLayer.getNeurons().length; i++) {
            Neuron neuron = outputLayer.getNeurons()[i];
            double error = target[i] - neuron.getOutput();
            double delta = error * neuron.getActivationFunction().derivative(neuron.getInputSum());
            neuron.setDelta(delta);
        }

        // 3. Вычисление ошибки для скрытых слоёв
        for (int l = layers.size() - 2; l >= 0; l--) { // от предпоследнего к первому
            Layer currentLayer = layers.get(l);
            Layer nextLayer = layers.get(l + 1);

            for (int i = 0; i < currentLayer.getNeurons().length; i++) {
                Neuron neuron = currentLayer.getNeurons()[i];
                double sum = 0.0;
                for (Neuron nextNeuron : nextLayer.getNeurons()) {
                    sum += nextNeuron.getWeights()[i] * nextNeuron.getDelta();
                }
                double delta = sum * neuron.getActivationFunction().derivative(neuron.getInputSum());
                neuron.setDelta(delta);
            }
        }

        // 4. Обновление весов и смещений
        double[] prevOutputs = input; // на входе первый слой получает input
        for (Layer layer : layers) {
            double[] newPrevOutputs = new double[layer.getNeurons().length];
            for (int i = 0; i < layer.getNeurons().length; i++) {
                Neuron neuron = layer.getNeurons()[i];
                double delta = neuron.getDelta();

                // Обновляем веса
                for (int j = 0; j < neuron.getWeights().length; j++) {
                    double oldWeight = neuron.getWeights()[j];
                    neuron.setWeight(j, oldWeight + learningRate * delta * prevOutputs[j]);
                }

                // Обновляем смещение
                neuron.setBias(neuron.getBias() + learningRate * delta);

                newPrevOutputs[i] = neuron.getOutput();
            }
            prevOutputs = newPrevOutputs;
        }
    }
}
