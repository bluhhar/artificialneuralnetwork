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
}
