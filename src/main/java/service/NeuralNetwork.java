package service;

import lombok.Getter;
import lombok.Setter;
import service.activationfunction.ActivationFunction;
import service.optimizer.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

@Getter
@Setter
public class NeuralNetwork {

    private List<Layer> layers;
    private Random random = new Random();
    private Optimizer optimizer;
    private double learningRate = 0.01;

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

    public void train(double[] input, double[] target) {
        predict(input);

        // 1. Вычисляем дельты на выходном слое
        Layer outputLayer = layers.getLast();
        for (int i = 0; i < outputLayer.getNeurons().length; i++) {
            Neuron neuron = outputLayer.getNeurons()[i];
            double error = target[i] - neuron.getOutput();
            double delta = error * neuron.getActivationFunction().derivative(neuron.getInputSum());
            neuron.setDelta(delta);
        }

        // 2. Вычисляем дельты для скрытых слоёв
        for (int l = layers.size() - 2; l >= 0; l--) {
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

        // 3. Обновляем веса через оптимизатор
        double[] prevOutputs = input;
        for (Layer layer : layers) {
            double[] newPrevOutputs = new double[layer.getNeurons().length];
            for (int i = 0; i < layer.getNeurons().length; i++) {
                Neuron neuron = layer.getNeurons()[i];

                optimizer.update(neuron, prevOutputs, learningRate); // 🔥 используем оптимизатор

                newPrevOutputs[i] = neuron.getOutput();
            }
            prevOutputs = newPrevOutputs;
        }
    }
}
