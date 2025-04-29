import entity.Iris;
import service.activationfunction.impl.Sigmoid;
import utility.IrisDataReader;

import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class Main {

    public static void main(String[] args) {
        //test1();
        //test2();
        //test3();
        //test4();
        //test5();
        test6();
    }

    private static void test1() {
        String resourcePath = "datasets/iris.csv";
        URL resourceUrl = Main.class.getClassLoader().getResource(resourcePath);
        Path path = null;
        try {
            path = Paths.get(resourceUrl.toURI());
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
        List<Iris> data = IrisDataReader.load(path.toString());
        System.out.println("Загружено " + data.size() + " образцов");
        System.out.println("Пример: " + Arrays.toString(data.getFirst().features) + " -> " + Arrays.toString(data.getFirst().label));
        for (Iris iris : data) {
            System.out.println(Arrays.toString(iris.features) + " -> " + Arrays.toString(iris.label));
        }
    }

    private static void test2() {
        NeuralNetwork network = new NeuralNetwork();
        network.addLayer(8, new Sigmoid()); // Скрытый слой на 8 нейронов
        network.addLayer(3, new Sigmoid()); // Выходной слой (3 класса)

        double[] prediction = network.predict(new double[]{0.5, 0.2, 0.7, 0.1});
        System.out.println(Arrays.toString(prediction));
    }

    private static void test3() {
        String resourcePath = "datasets/iris.csv";
        URL resourceUrl = Main.class.getClassLoader().getResource(resourcePath);
        Path path = null;
        try {
            path = Paths.get(resourceUrl.toURI());
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
        List<Iris> dataset = IrisDataReader.load(path.toString());

        NeuralNetwork network = new NeuralNetwork();
        network.addLayer(8, new Sigmoid());
        network.addLayer(3, new Sigmoid());

        double learningRate = 0.1;
        int epochs = 1000;

        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0.0;
            for (Iris sample : dataset) {
                network.train(sample.features, sample.label);

                // Вычисляем ошибку (сумма квадратов ошибок)
                double[] output = network.predict(sample.features);
                for (int i = 0; i < output.length; i++) {
                    totalError += Math.pow(sample.label[i] - output[i], 2);
                }
            }
            if (epoch % 100 == 0) {
                System.out.println("Эпоха " + epoch + ", Ошибка: " + totalError);
            }
        }
    }

    private static void test4() {
        String resourcePath = "datasets/iris.csv";
        URL resourceUrl = Main.class.getClassLoader().getResource(resourcePath);
        Path path = null;
        try {
            path = Paths.get(resourceUrl.toURI());
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
        List<Iris> dataset = IrisDataReader.load(path.toString());

        NeuralNetwork network = new NeuralNetwork();
        network.addLayer(8, new Sigmoid());
        network.addLayer(3, new Sigmoid());
        network.setOptimizer(new SGDOptimizer());
        network.setLearningRate(0.1);

        for (Iris sample : dataset) {
            network.train(sample.features, sample.label);
        }
    }

    private static void test5() {
        String resourcePath = "datasets/iris.csv";
        URL resourceUrl = Main.class.getClassLoader().getResource(resourcePath);
        Path path = null;
        try {
            path = Paths.get(resourceUrl.toURI());
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
        List<Iris> dataset = IrisDataReader.load(path.toString());

        NeuralNetwork network = new NeuralNetwork();
        network.addLayer(8, new Sigmoid());
        network.addLayer(3, new Sigmoid());
        network.setOptimizer(new AMSGradOptimizer());
        network.setLearningRate(0.01);

        int epochs = 1000;
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (Iris sample : dataset) {
                network.train(sample.features, sample.label);
            }
        }
    }

    private static void test6() {
        // 1. Загружаем данные
        String resourcePath = "datasets/iris.csv";
        URL resourceUrl = Main.class.getClassLoader().getResource(resourcePath);
        Path path = null;
        try {
            path = Paths.get(resourceUrl.toURI());
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
        List<Iris> dataset = IrisDataReader.load(path.toString());

        // 2. Инициализируем сеть
        NeuralNetwork network = new NeuralNetwork();
        network.addLayer(8, new Sigmoid());
        network.addLayer(3, new Sigmoid());
        network.setOptimizer(new AMSGradOptimizer());
        network.setLearningRate(0.01);

        // 3. Параметры обучения
        int epochs = 500;
        Random random = new Random();

        for (int epoch = 1; epoch <= epochs; epoch++) {
            Collections.shuffle(dataset, random); // Перемешиваем для обучения

            double totalLoss = 0.0;
            int correct = 0;

            for (Iris sample : dataset) {
                network.train(sample.features, sample.label);

                double[] prediction = network.predict(sample.features);

                // Подсчёт ошибки (сумма квадратов ошибок)
                for (int i = 0; i < prediction.length; i++) {
                    totalLoss += Math.pow(sample.label[i] - prediction[i], 2);
                }

                // Подсчёт правильных предсказаний
                if (argMax(prediction) == argMax(sample.label)) {
                    correct++;
                }
            }

            double averageLoss = totalLoss / dataset.size();
            double accuracy = (correct * 100.0) / dataset.size();

            if (epoch % 10 == 0 || epoch == 1) {
                System.out.printf("Эпоха %d: Loss = %.6f, Accuracy = %.2f%%\n", epoch, averageLoss, accuracy);
            }
        }
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
