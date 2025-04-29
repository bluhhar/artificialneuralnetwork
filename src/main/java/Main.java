import activationfunction.impl.ReLU;
import activationfunction.impl.Sigmoid;

import java.util.Arrays;

public class Main {

    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork();
        nn.addLayer(new Layer(5, 4, new ReLU()));
        nn.addLayer(new Layer(3, 5, new Sigmoid()));

        double[] input = {5.1, 3.5, 1.4, 0.2};
        double[] output = nn.predict(input);

        System.out.println("Output: " + Arrays.toString(output));
    }

}
