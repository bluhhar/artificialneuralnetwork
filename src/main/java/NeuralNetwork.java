import lombok.Getter;
import lombok.Setter;
import service.activationfunction.ActivationFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

@Getter
@Setter
public class NeuralNetwork {

    private List<Layer> layers;
    private Random random = new Random();

    public NeuralNetwork() {
        layers = new ArrayList<>();
    }

    public void addLayer(int numberOfNeurons, ActivationFunction activationFunction) {
        int inputSize = layers.isEmpty() ? 4 : layers.get(layers.size() - 1).getNeurons().length;
        layers.add(new Layer(numberOfNeurons, inputSize, activationFunction, random));
    }

    public double[] predict(double[] input) {
        double[] output = input;
        for (Layer layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }
}
