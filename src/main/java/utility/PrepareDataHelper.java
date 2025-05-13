package utility;

import service.ConvolutionalNeuralNetwork;
import service.Pair;

import java.io.*;
import java.util.*;

public class PrepareDataHelper {

    public static List<Pair<double[][], double[]>> convertToCnnFormat(List<Pair<List<Double>, List<Double>>> imageData) {
        List<Pair<double[][], double[]>> result = new ArrayList<>();

        int imageSize = (int) Math.sqrt(imageData.getFirst().first().size());

        for (Pair<List<Double>, List<Double>> pair : imageData) {
            List<Double> pixels = pair.first();
            List<Double> targets = pair.second();

            double[][] image = new double[imageSize][imageSize];
            for (int i = 0; i < imageSize; i++) {
                for (int j = 0; j < imageSize; j++) {
                    image[i][j] = pixels.get(i * imageSize + j);
                }
            }

            double[] targetArray = targets.stream().mapToDouble(Double::doubleValue).toArray();

            result.add(new Pair<>(image, targetArray));
        }

        return result;
    }

    /**
     * Сохранение оригинальной модели в бинарный файл
     * @param model Модель
     * @param path путь к модели
     */
    public static void saveOriginalModel(ConvolutionalNeuralNetwork model, String path) {
        try (DataOutputStream writer = new DataOutputStream(new FileOutputStream(path))) {
            writer.writeBytes("ORIG"); // сигнатура
            writer.writeInt(1); // версия

            List<double[][]> convKernels = model.convLayer.getKernels();
            writer.writeInt(convKernels.size());
            for (double[][] kernel : convKernels) {
                for (double[] row : kernel) {
                    for (double val : row) {
                        writer.writeDouble(val);
                    }
                }
            }

            double[][] fcWeights = model.fcLayer.getWeights();
            for (double[] row : fcWeights) {
                for (double val : row) {
                    writer.writeDouble(val);
                }
            }

            System.out.println("Модель сохранена в " + path + " (оригинальный формат)");

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Сохранение квантизованной модели в 8-битном формате
     * @param model Модель
     * @param path путь к модели
      */
    public static void saveQuantizedModel(ConvolutionalNeuralNetwork model, String path) {
        try (DataOutputStream writer = new DataOutputStream(new FileOutputStream(path))) {
            writer.writeBytes("QNT"); // сигнатура
            writer.writeInt(1);       // версия

            List<double[][]> convKernels = model.convLayer.getKernels();
            writer.writeInt(convKernels.size());
            for (double[][] kernel : convKernels) {
                for (double[] row : kernel) {
                    for (double val : row) {
                        writer.writeByte((int) (val * 255));
                    }
                }
            }

            double[][] fcWeights = model.fcLayer.getWeights();
            for (double[] row : fcWeights) {
                for (double val : row) {
                    writer.writeByte((int) (val * 255));
                }
            }

            System.out.println("Модель сохранена в " + path + " (8-битный формат)");

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
