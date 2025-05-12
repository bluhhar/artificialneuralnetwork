package service;

import entity.Iris;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import service.activationfunction.ActivationFunction;
import service.activationfunction.impl.Sigmoid;
import service.optimizer.Optimizer;
import service.optimizer.impl.SGDOptimizer;
import util.Fixtures;
import util.Timer;
import utility.IrisDataReader;

import java.util.Collections;
import java.util.List;
import java.util.Random;

public class JordanNeuralNetworkTest {

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

        int inputSize = 1;          // один признак на временной шаг
        int hiddenSize = 8;
        int outputSize = 3;         // 3 класса
        double learningRate = 0.01;

        JordanNetwork model = new JordanNetwork(inputSize, hiddenSize, outputSize, 4);

        int epochs = 300;

        for (int epoch = 1; epoch <= epochs; epoch++) {
            Collections.shuffle(trainSet, new Random());

            double totalLoss = 0.0;
            int correct = 0;

            for (Iris sample : trainSet) {
                model.train(sample, learningRate);

                double[] prediction = model.predict(toSequence(sample.features));
                for (int i = 0; i < prediction.length; i++) {
                    totalLoss += Math.pow(sample.label[i] - prediction[i], 2);
                }

                if (argMax(prediction) == argMax(sample.label)) {
                    correct++;
                }
            }

            double trainAccuracy = correct * 100.0 / trainSet.size();
            if (epoch == 1 || epoch % 10 == 0) {
                double testAccuracy = evaluate(model, testSet);
                System.out.printf("Эпоха %d: Train Accuracy = %.2f%%, Test Accuracy = %.2f%%, Loss = %.4f\n",
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

    private static int argMax(double[] array) {
        int index = 0;
        double max = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
                index = i;
            }
        }
        return index;
    }

    private static double evaluate(JordanNetwork model, List<Iris> testSet) {
        int correct = 0;
        for (Iris sample : testSet) {
            double[] prediction = model.predict(toSequence(sample.features));
            if (argMax(prediction) == argMax(sample.label)) {
                correct++;
            }
        }
        return correct * 100.0 / testSet.size();
    }
}
