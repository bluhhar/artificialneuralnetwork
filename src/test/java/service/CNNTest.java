package service;

import org.junit.jupiter.api.Test;
import utility.ImagesReader;
import utility.PrepareDataHelper;

import java.util.*;

import static util.Fixtures.getModelsPath;

public class CNNTest {

    @Test
    void testBaseCnn() {
        var trainRaw = ImagesReader.getPreparedData(ImagesReader.TypeData.TRAIN);
        var trainSet = PrepareDataHelper.convertToCnnFormat(trainRaw);
        var testRaw = ImagesReader.getPreparedData(ImagesReader.TypeData.TEST);
        var testSet = PrepareDataHelper.convertToCnnFormat(testRaw);

        var cnn = new ConvolutionalNeuralNetwork(
                28,
                28,
                8,
                16,
                10,
                0.001,
                false,
                false,
                false
        );

        long start = System.currentTimeMillis();
        trainModel(cnn, trainSet, trainSet, 1000, 30);
        testModel(cnn, testSet);
        System.out.printf("Время вывода без прунинга: %.2f сек\n", (System.currentTimeMillis() - start) / 1000.0);
    }

    @Test
    void testQuantize() {
        var trainSet = PrepareDataHelper.convertToCnnFormat(ImagesReader.getPreparedData(ImagesReader.TypeData.TRAIN));
        var testSet = PrepareDataHelper.convertToCnnFormat(ImagesReader.getPreparedData(ImagesReader.TypeData.TEST));

        var cnn = new ConvolutionalNeuralNetwork(
                28,
                28,
                8,
                16,
                10,
                0.001,
                false,
                true,
                false
        );

        long start = System.currentTimeMillis();
        trainModel(cnn, trainSet, trainSet, 1000, 30);
        System.out.printf("Время обучения: %.2f сек\n", (System.currentTimeMillis() - start) / 1000.0);

        System.out.println("\nТестирование ДО квантизации:");
        testModel(cnn, testSet);

        PrepareDataHelper.saveOriginalModel(cnn, getModelsPath() + "/model.bin");
        cnn.quantizeModel();

        System.out.println("\nТестирование ПОСЛЕ квантизации:");
        testModel(cnn, testSet);
        PrepareDataHelper.saveQuantizedModel(cnn, getModelsPath() + "/quantized_model.bin");
    }

    public static void trainModel(ConvolutionalNeuralNetwork cnn,
                                  List<ImagesReader.Pair<double[][], double[]>> trainSet,
                                  List<ImagesReader.Pair<double[][], double[]>> valSet,
                                  int epochs, int batchSize) {
        double bestAccuracy = 0;
        int patience = 4, epochsWithoutImprovement = 0;
        double minDelta = 0.001;

        for (int epoch = 1; epoch <= epochs; epoch++) {
            PrepareDataHelper.shuffle(trainSet);
            double totalLoss = 0;
            int batchCount = 0;

            for (int i = 0; i < trainSet.size(); i += batchSize) {
                var batch = trainSet.subList(i, Math.min(trainSet.size(), i + batchSize));
                for (var sample : batch) {
                    var output = cnn.forward(sample.first);
                    var loss = cnn.calculateLoss(output, sample.second);
                    totalLoss += loss;
                    cnn.train(sample.first, sample.second, null);
                }

                if (++batchCount % 10 == 0) {
                    System.out.printf("Эпоха %d, Пакет %d, Средний Loss: %.4f\n", epoch, batchCount, totalLoss / batchCount);
                }
            }

            double acc = cnn.evaluate(valSet);
            System.out.printf("Эпоха %d завершена. Точность: %.2f%%\n", epoch, acc * 100);

            if (epoch % 5 == 0 && cnn.enablePruning) {
                cnn.prune(cnn.pruningSparsity);
            }

            if (acc > bestAccuracy + minDelta) {
                bestAccuracy = acc;
                epochsWithoutImprovement = 0;
            } else {
                epochsWithoutImprovement++;
                System.out.printf("Точность не улучшилась %d/%d\n", epochsWithoutImprovement, patience);
                if (epochsWithoutImprovement >= patience) {
                    System.out.println("Early stopping!");
                    break;
                }
            }
        }
    }

    public static void testModel(ConvolutionalNeuralNetwork cnn,
                                 List<ImagesReader.Pair<double[][], double[]>> testSet) {
        int[][] confusion = new int[10][10];
        double[] avgProb = new double[10];
        int[] classCount = new int[10];

        for (var sample : testSet) {
            var output = cnn.forward(sample.first);
            int predicted = argMax(output);
            int actual = argMax(sample.second);
            confusion[actual][predicted]++;
            avgProb[actual] += output[actual];
            classCount[actual]++;
        }

        int correct = 0;
        for (int i = 0; i < 10; i++) {
            correct += confusion[i][i];
        }

        double accuracy = (double) correct / testSet.size();
        System.out.printf("Финальная точность: %d/%d (%.2f%%)\n", correct, testSet.size(), accuracy * 100);

        System.out.println("\nКласс | Правильно | Всего | Точность | Ср. вероятность");
        for (int i = 0; i < 10; i++) {
            int total = classCount[i];
            double classAcc = total > 0 ? (double) confusion[i][i] / total : 0;
            double avg = total > 0 ? avgProb[i] / total : 0;
            System.out.printf("  %d   |    %4d    | %4d |   %5.1f%% |    %5.2f\n", i, confusion[i][i], total, classAcc * 100, avg);
        }

        System.out.println("\nМатрица ошибок:");
        for (int i = 0; i < 10; i++) {
            System.out.printf("%2d |", i);
            for (int j = 0; j < 10; j++) {
                System.out.printf("%4d", confusion[i][j]);
            }
            System.out.println();
        }
    }

    private static int argMax(double[] arr) {
        int idx = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[idx]) idx = i;
        }
        return idx;
    }
}
