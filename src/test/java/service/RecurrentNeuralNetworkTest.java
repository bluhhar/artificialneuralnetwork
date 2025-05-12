package service;

import entity.Iris;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import service.activationfunction.ActivationFunction;
import service.activationfunction.impl.Sigmoid;
import service.optimizer.Optimizer;
import service.optimizer.impl.SGDOptimizer;
import util.Exec;
import util.Fixtures;
import util.Timer;
import utility.IrisDataReader;

import java.util.Collections;
import java.util.List;
import java.util.Random;

public class RecurrentNeuralNetworkTest {

    private Fixtures fixtures;
    private Timer timer;
    private List<Iris> dataset;

    @BeforeEach
    void setUp() {
        fixtures = new Fixtures();
        dataset = IrisDataReader.load(fixtures.getRecoursePath());
    }

    @Test
    public void testRecurrentNeuralNetwork() {
        Collections.shuffle(dataset, new Random());

        int trainSize = (int) (dataset.size() * 0.8);
        List<Iris> trainSet = dataset.subList(0, trainSize);
        List<Iris> testSet = dataset.subList(trainSize, dataset.size());

        System.out.println("Обучающая выборка: " + trainSet.size() + " примеров");
        System.out.println("Тестовая выборка: " + testSet.size() + " примеров");

        ActivationFunction activationFunction = new Sigmoid();
        Optimizer optimizer = new SGDOptimizer();
        double learningRate = 0.01;

        RecurrentNeuralNetwork rnn = new RecurrentNeuralNetwork(
                1,          // input size per timestep (один признак)
                12,                  // размер скрытого состояния LSTM
                3,                  // выходной слой — 3 класса
                activationFunction,
                optimizer,
                learningRate
        );

        int epochs = 10000;

        for (int epoch = 1; epoch <= epochs; epoch++) {
            Collections.shuffle(trainSet, new Random());

            double totalLoss = 0.0;
            int correct = 0;

            for (Iris sample : trainSet) {
                rnn.train(sample);

                double[] prediction = rnn.predict(toSequence(sample.features));
                for (int i = 0; i < prediction.length; i++) {
                    totalLoss += Math.pow(sample.label[i] - prediction[i], 2);
                }

                if (argMax(prediction) == argMax(sample.label)) {
                    correct++;
                }
            }

            double trainAccuracy = (correct * 100.0) / trainSet.size();
            if (epoch % 100 == 0 || epoch == 1) {
                double testAccuracy = evaluate(rnn, testSet);
                System.out.printf("Эпоха %d: Train Accuracy = %.2f%%, Test Accuracy = %.2f%%, Total Loss = %.2f\n",
                        epoch, trainAccuracy, testAccuracy, totalLoss);
            }
        }
    }


    private static double[][] toSequence(double[] features) {
        double[][] seq = new double[features.length][1];
        for (int i = 0; i < features.length; i++) {
            seq[i][0] = features[i];
        }
        return seq;
    }

    private static double evaluate(RecurrentNeuralNetwork rnn, List<Iris> testSet) {
        int correct = 0;
        for (Iris sample : testSet) {
            double[] prediction = rnn.predict(toSequence(sample.features));
            if (argMax(prediction) == argMax(sample.label)) {
                correct++;
            }
        }
        return (correct * 100.0) / testSet.size();
    }

    private static int argMax(double[] array) {
        int bestIndex = 0;
        double bestValue = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > bestValue) {
                bestValue = array[i];
                bestIndex = i;
            }
        }
        return bestIndex;
    }
}
