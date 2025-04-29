import entity.Iris;
import utility.IrisDataReader;

import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

public class Main {

    public static void main(String[] args) {
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

}
