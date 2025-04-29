package utility;

import entity.Iris;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class IrisDataReader {

    public static List<Iris> load(String filePath) {
        List<Iris> samples = new ArrayList<>();
        List<double[]> featuresList = new ArrayList<>();
        List<String> labelsRaw = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            br.readLine();
            while ((line = br.readLine()) != null) {
                String[] tokens = line.split(",");
                double[] features = new double[4];
                for (int i = 0; i < 4; i++) {
                    features[i] = Double.parseDouble(tokens[i]);
                }
                featuresList.add(features);
                labelsRaw.add(tokens[4]);
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        double[][] normalizedFeatures = normalize(featuresList);
        for (int i = 0; i < normalizedFeatures.length; i++) {
            samples.add(new Iris(normalizedFeatures[i], toLabel(labelsRaw.get(i))));
        }

        return samples;
    }

    private static double[][] normalize(List<double[]> featuresList) {
        int numFeatures = featuresList.getFirst().length;
        int numSamples = featuresList.size();
        double[][] features = new double[numSamples][numFeatures];

        double[] min = new double[numFeatures];
        double[] max = new double[numFeatures];
        Arrays.fill(min, Double.POSITIVE_INFINITY);
        Arrays.fill(max, Double.NEGATIVE_INFINITY);

        for (int i = 0; i < numSamples; i++) {
            double[] f = featuresList.get(i);
            for (int j = 0; j < numFeatures; j++) {
                min[j] = Math.min(min[j], f[j]);
                max[j] = Math.max(max[j], f[j]);
            }
        }

        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                features[i][j] = (featuresList.get(i)[j] - min[j]) / (max[j] - min[j]);
            }
        }

        return features;
    }

    private static double[] toLabel(String label) {
        String cleanLabel = label.replace("\"", "");
        return switch (cleanLabel) {
            case "Setosa" -> new double[]{1, 0, 0};
            case "Versicolor" -> new double[]{0, 1, 0};
            case "Virginica" -> new double[]{0, 0, 1};
            default -> throw new IllegalArgumentException("Unknown label: " + label);
        };
    }
}
