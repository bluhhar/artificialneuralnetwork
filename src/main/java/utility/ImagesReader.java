package utility;

import service.Pair;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class ImagesReader {
    public enum TypeData {
        TRAIN, TEST
    }

    public static List<Pair<List<Double>, List<Double>>> getPreparedData(TypeData typeData) {
        String path = getImagesPath();
        if (typeData == TypeData.TRAIN) {
            path += "/train/";
        } else {
            path += "/test/";
        }

        List<Pair<List<Double>, List<Double>>> trainingData = new ArrayList<>();

        for (int i = 0; i < 10; i++) {
            File dir = new File(path + i);
            File[] files = dir.listFiles();
            if (files == null) continue;

            List<Double> outputs = getDesiredOutputs(i);
            int k = 0;
            for (File file : files) {
                if (k > 100) break;
                try {
                    trainingData.add(new Pair<>(convertImageToFunctionSignal(file), outputs));
                } catch (IOException e) {
                    e.printStackTrace();
                }
                k++;
            }
        }

        return trainingData;
    }

    public static List<Double> convertImageToFunctionSignal(File file) throws IOException {
        List<Double> functionSignal = new ArrayList<>();
        BufferedImage img = ImageIO.read(file);

        for (int i = 0; i < img.getWidth(); i++) {
            for (int j = 0; j < img.getHeight(); j++) {
                int pixel = img.getRGB(i, j);
                // Если пиксель белый (0xFFFFFFFF)
                if (pixel == 0xFFFFFFFF) {
                    functionSignal.add(0.0);
                } else {
                    functionSignal.add(1.0);
                }
            }
        }

        return functionSignal;
    }

    public static List<Double> getDesiredOutputs(int num) {
        List<Double> outputs = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            outputs.add(i == num ? 1.0 : 0.0);
        }
        return outputs;
    }

    public static String getImagesPath() {
        String path = "src/main/resources/datasets/MNIST/";

        File file = new File(path);
        String absolutePath = file.getAbsolutePath();

        return absolutePath;
    }
}
