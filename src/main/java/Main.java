import entity.Iris;
import service.NeuralNetwork;
import service.activationfunction.ActivationFunction;
import service.activationfunction.impl.Sigmoid;
import service.optimizer.Optimizer;
import utility.IrisDataReader;

import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class Main {

    private static List<Iris> getIrisDataset() {
        String resourcePath = "datasets/iris.csv";
        URL resourceUrl = Main.class.getClassLoader().getResource(resourcePath);
        Path path = null;
        try {
            path = Paths.get(resourceUrl.toURI());
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
        return IrisDataReader.load(path.toString());
    }

    public static void main(String[] args) {
        //testOptimizer(new service.optimizer.impl.AMSGradOptimizer(), new Sigmoid(), 0.01);
        //testOptimizer(new service.optimizer.impl.RpropOptimizer(), 0);
        //testOptimizer(new service.optimizer.impl.RmspropGravesOptimizer(), 0.01);
        //testOptimizer(new service.optimizer.impl.QuickpropOptimizer(), new ReLU(), 0.01);
        //testOptimizer(new service.optimizer.impl.SGDOptimizer(new service.regularizer.impl.GroupLassoRegularizer(0.001)), new Sigmoid(), 0.01);
        //testOptimizer(new service.optimizer.impl.AMSGradOptimizer(new service.regularizer.impl.GroupLassoRegularizer(0.001)), new Sigmoid(), 0.01);
        testOptimizer(new service.optimizer.impl.RmspropGravesOptimizer(new service.regularizer.impl.GroupLassoRegularizer(0.001)), new Sigmoid(), 0.01);
    }

    private static void testOptimizer(Optimizer optimizer, ActivationFunction activationFunction, double learningRate) {
        // 1. Загружаем данные
        List<Iris> dataset = getIrisDataset();

        // 2. Перемешиваем данные
        Collections.shuffle(dataset, new Random());

        // 3. Делим на обучающую и тестовую выборки
        int trainSize = (int) (dataset.size() * 0.8);
        List<Iris> trainSet = dataset.subList(0, trainSize);
        List<Iris> testSet = dataset.subList(trainSize, dataset.size());

        System.out.println("Обучающая выборка: " + trainSet.size() + " примеров");
        System.out.println("Тестовая выборка: " + testSet.size() + " примеров");

        // 4. Инициализируем сеть
        NeuralNetwork network = new NeuralNetwork();
        network.addLayer(8, activationFunction);
        network.addLayer(3, activationFunction);
        network.setOptimizer(optimizer);
        network.setLearningRate(learningRate);

        // 5. Параметры обучения
        int epochs = 500;
        Random random = new Random();

        for (int epoch = 1; epoch <= epochs; epoch++) {
            Collections.shuffle(trainSet, random); // Перемешиваем только тренировку

            double totalLoss = 0.0;
            int correct = 0;

            for (Iris sample : trainSet) {
                network.train(sample.features, sample.label);

                double[] prediction = network.predict(sample.features);

                // Вычисляем ошибку
                for (int i = 0; i < prediction.length; i++) {
                    totalLoss += Math.pow(sample.label[i] - prediction[i], 2);
                }

                // Подсчёт правильных предсказаний на обучении
                if (argMax(prediction) == argMax(sample.label)) {
                    correct++;
                }
            }

            double averageLoss = totalLoss / trainSet.size();
            double trainAccuracy = (correct * 100.0) / trainSet.size();

            if (epoch % 10 == 0 || epoch == 1) {
                double testAccuracy = evaluate(network, testSet);
                System.out.printf("Эпоха %d: Train Loss = %.6f, Train Accuracy = %.2f%%, Test Accuracy = %.2f%%\n",
                        epoch, averageLoss, trainAccuracy, testAccuracy);
            }
        }
    }

    private static double evaluate(NeuralNetwork network, List<Iris> testSet) {
        int correct = 0;
        for (Iris sample : testSet) {
            double[] prediction = network.predict(sample.features);
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
