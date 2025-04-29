import entity.Iris;
import service.activationfunction.impl.Sigmoid;
import utility.IrisDataReader;

import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

public class Main {

    public static void main(String[] args) {
        //test1();
        test2();
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
}
