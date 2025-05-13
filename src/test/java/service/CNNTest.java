package service;

import org.junit.jupiter.api.Test;
import utility.ImagesReader;
import utility.PrepareDataHelper;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class CNNTest {

    @Test
    void testCNN() {
        var imageTrainData = ImagesReader.getPreparedData(ImagesReader.TypeData.TRAIN);
        var trainingData = PrepareDataHelper.convertToCnnFormat(imageTrainData);
        var imageTestData = ImagesReader.getPreparedData(ImagesReader.TypeData.TEST);
        var testingData = PrepareDataHelper.convertToCnnFormat(imageTestData);

        // 2. Инициализация модели
        var cnn = new ConvolutionalNeuralNetwork(
                28, 28,
                5,       // kernelSize
                16,      // numKernels
                10,      // fcOutputSize
                0.0001   // learningRate
        );

        long start = System.nanoTime();

        // 3. Обучение
        trainModel(cnn, trainingData, trainingData, 1000, 30);

        double elapsed = (System.nanoTime() - start) / 1e9;
        System.out.printf("Время обучения: %.2f сек\n", elapsed);

        // 4. Тестирование
        testModel(cnn, testingData);
    }


    public static void trainModel(ConvolutionalNeuralNetwork cnn,
                                  List<Pair<double[][], double[]>> trainSet,
                                  List<Pair<double[][], double[]>> valSet,
                                  int epochs, int batchSize,
                                  int patience, double minDelta) {
        System.out.println("Начинаем обучение...");

        double bestAccuracy = 0;
        int epochsWithoutImprovement = 0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;
            int batchCount = 0;

            Collections.shuffle(trainSet);

            for (int i = 0; i < trainSet.size(); i += batchSize) {
                var batch = trainSet.subList(i, Math.min(i + batchSize, trainSet.size()));

                for (var sample : batch) {
                    cnn.train(sample.first(), sample.second());
                    var output = cnn.forward(sample.first());
                    totalLoss += cnn.calculateLoss(output, sample.second());
                }

                if (++batchCount % 10 == 0) {
                    System.out.printf("Эпоха: %d, Пакет: %d, Loss: %.4f\n",
                            epoch + 1, batchCount, totalLoss / (batchCount * batchSize));
                }
            }

            double acc = cnn.evaluate(valSet);
            System.out.printf("Эпоха %d завершена. Точность: %.2f%%\n", epoch + 1, acc * 100);

            if (acc > bestAccuracy + minDelta) {
                bestAccuracy = acc;
                epochsWithoutImprovement = 0;
            } else {
                epochsWithoutImprovement++;
                System.out.printf("Точность не улучшилась %d/%d\n", epochsWithoutImprovement, patience);
                if (epochsWithoutImprovement >= patience) {
                    System.out.println("Early Stopping!");
                    break;
                }
            }

            System.out.println();
        }
    }

    public static void trainModel(ConvolutionalNeuralNetwork cnn,
                                  List<Pair<double[][], double[]>> trainSet,
                                  List<Pair<double[][], double[]>> valSet,
                                  int epochs, int batchSize) {
        trainModel(cnn, trainSet, valSet, epochs, batchSize, 3, 0.001);
    }

    public static void testModel(ConvolutionalNeuralNetwork cnn,
                                 List<Pair<double[][], double[]>> testSet) {
        System.out.println("\n=== Детальное тестирование модели ===");

        int numClasses = 10;
        int[][] confusion = new int[numClasses][numClasses];
        double[] classProbs = new double[numClasses];
        int[] classCounts = new int[numClasses];

        for (var sample : testSet) {
            double[] output = cnn.forward(sample.first());
            int predicted = argMax(output);
            int actual = argMax(sample.second());

            confusion[actual][predicted]++;
            classProbs[actual] += output[actual];
            classCounts[actual]++;
        }

        int correct = 0;
        for (int i = 0; i < numClasses; i++) {
            correct += confusion[i][i];
        }

        double accuracy = (double) correct / testSet.size();
        System.out.printf("\nФинальная точность: %d/%d (%.2f%%)\n", correct, testSet.size(), accuracy * 100);

        System.out.println("\nКласс | Правильно | Всего | Точность | Ср. вероятность");
        System.out.println("------|-----------|-------|----------|----------------");

        for (int i = 0; i < numClasses; i++) {
            int tp = confusion[i][i];
            int total = classCounts[i];
            double classAcc = total > 0 ? (double) tp / total : 0;
            double avgProb = total > 0 ? classProbs[i] / total : 0;
            System.out.printf(" %4d | %8d | %5d | %8.1f%% | %14.2f\n", i, tp, total, classAcc * 100, avgProb);
        }

        System.out.println("\nМатрица ошибок:");
        System.out.print("     ");
        for (int j = 0; j < numClasses; j++) System.out.printf("%5d", j);
        System.out.println("\n     " + "-".repeat(5 * numClasses));

        for (int i = 0; i < numClasses; i++) {
            System.out.printf("%4d|", i);
            for (int j = 0; j < numClasses; j++) {
                System.out.printf("%5d", confusion[i][j]);
            }
            System.out.println();
        }

        System.out.println("\nТипичные ошибки:");
        List<ErrorCase> errors = new ArrayList<>();
        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < numClasses; j++) {
                if (i != j && confusion[i][j] > 0) {
                    errors.add(new ErrorCase(i, j, confusion[i][j]));
                }
            }
        }

        errors.stream()
                .sorted((a, b) -> Integer.compare(b.count, a.count))
                .limit(5)
                .forEach(e -> System.out.printf("%d раз: %d → %d\n", e.count, e.actual, e.predicted));
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

    private static class ErrorCase {
        int actual, predicted, count;

        public ErrorCase(int actual, int predicted, int count) {
            this.actual = actual;
            this.predicted = predicted;
            this.count = count;
        }
    }
}
