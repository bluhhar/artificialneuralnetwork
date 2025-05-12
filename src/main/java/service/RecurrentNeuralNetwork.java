package service;

import entity.Iris;
import service.activationfunction.ActivationFunction;
import service.optimizer.Optimizer;

import java.util.Random;

public class RecurrentNeuralNetwork {
    private final LSTMLayer lstmLayer;
    private final Layer outputLayer;
    private final Optimizer optimizer;
    private final double learningRate;

    public RecurrentNeuralNetwork(int inputSize, int lstmHiddenSize, int outputSize,
                                  ActivationFunction activationFunction,
                                  Optimizer optimizer,
                                  double learningRate) {

        this.lstmLayer = new LSTMLayer(inputSize, lstmHiddenSize);
        this.outputLayer = new Layer(outputSize, lstmHiddenSize, activationFunction, new Random());
        this.optimizer = optimizer;
        this.learningRate = learningRate;
    }

    public double[] predict(double[][] inputSequence) {
        double[] lstmOutput = lstmLayer.forward(inputSequence);
        return outputLayer.forward(lstmOutput);
    }

    public void train(Iris sample) {
        double[] lstmOutput = lstmLayer.forward(toSequence(sample.features));
        double[] prediction = outputLayer.forward(lstmOutput);

        // Вычисляем дельты выходного слоя
        for (int i = 0; i < outputLayer.getNeurons().length; i++) {
            var neuron = outputLayer.getNeurons()[i];
            double error = sample.label[i] - prediction[i];
            double delta = error * neuron.getActivationFunction().derivative(neuron.getInputSum());
            neuron.setDelta(delta);
        }

        // Обновляем веса выходного слоя
        for (int i = 0; i < outputLayer.getNeurons().length; i++) {
            var neuron = outputLayer.getNeurons()[i];
            optimizer.update(neuron, lstmOutput, learningRate);
        }
    }

    private double[][] toSequence(double[] features) {
        // Преобразуем 4 признака → 4 временных шага с 1 входом
        double[][] sequence = new double[features.length][1];
        for (int i = 0; i < features.length; i++) {
            sequence[i][0] = features[i];
        }
        return sequence;
    }
}
