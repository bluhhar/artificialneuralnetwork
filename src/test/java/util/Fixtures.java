package util;

import java.io.File;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Fixtures {

    public String getRecoursePath() {
        String resourcePath = "datasets/iris.csv";
        URL resourceUrl = Fixtures.class.getClassLoader().getResource(resourcePath);
        Path path;
        try {
            if (resourceUrl != null) {
                path = Paths.get(resourceUrl.toURI());
            } else {
                throw new RuntimeException();
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return path.toString();
    }

    public static String getModelsPath() {
        String path = "src/main/resources/models/";

        File file = new File(path);
        String absolutePath = file.getAbsolutePath();

        return absolutePath;
    }
}
