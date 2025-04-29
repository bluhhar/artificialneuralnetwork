package util;

import entity.Iris;
import lombok.extern.slf4j.Slf4j;
import service.NeuralNetwork;
import service.activationfunction.ActivationFunction;
import service.optimizer.Optimizer;

import java.util.Collections;
import java.util.List;
import java.util.Random;

@Slf4j
public class Exec {

    public boolean test(List<Iris> dataset,
                        Optimizer optimizer,
                        ActivationFunction activationFunction,
                        double learningRate,
                        int epochs,
                        String label,
                        boolean isPrintResult) {

        // 1. Перемешиваем данные
        Collections.shuffle(dataset, new Random());

        // 2. Делим на обучающую и тестовую выборки
        int trainSize = (int) (dataset.size() * 0.8);
        List<Iris> trainSet = dataset.subList(0, trainSize);
        List<Iris> testSet = dataset.subList(trainSize, dataset.size());

        if (isPrintResult) {
            log.info("Optimizer {} dataset preview trainSet size {}, testSet size {}",
                    label, trainSet.size(), testSet.size());
        }

        // 3. Инициализируем сеть
        NeuralNetwork network = new NeuralNetwork();
        network.addLayer(8, activationFunction);
        network.addLayer(3, activationFunction);
        network.setOptimizer(optimizer);
        network.setLearningRate(learningRate);

        Random random = new Random();

        for (int epoch = 1; epoch <= epochs; epoch++) {
            Collections.shuffle(trainSet, random);

            double totalLoss = 0.0;
            int correct = 0;

            for (Iris sample : trainSet) {
                network.train(sample.features, sample.label);

                double[] prediction = network.predict(sample.features);

                //вычисляем ошибку
                for (int i = 0; i < prediction.length; i++) {
                    totalLoss += Math.pow(sample.label[i] - prediction[i], 2);
                }

                //подсчёт правильных предсказаний на обучении
                if (argMax(prediction) == argMax(sample.label)) {
                    correct++;
                }
            }

            double averageLoss = totalLoss / trainSet.size();
            double trainAccuracy = (correct * 100.0) / trainSet.size();

            if (epoch % 10 == 0 || epoch == 1) {
                double testAccuracy = evaluate(network, testSet);
                if (isPrintResult) {
                    System.out.printf("epoch %d: Train Loss = %.6f, Train Accuracy = %.2f%%, Test Accuracy = %.2f%%\n",
                            epoch, averageLoss, trainAccuracy, testAccuracy);
                }
            }
        }
        return true;
    }

    private double evaluate(NeuralNetwork network, List<Iris> testSet) {
        int correct = 0;
        for (Iris sample : testSet) {
            double[] prediction = network.predict(sample.features);
            if (argMax(prediction) == argMax(sample.label)) {
                correct++;
            }
        }
        return (correct * 100.0) / testSet.size();
    }

    private int argMax(double[] array) {
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
