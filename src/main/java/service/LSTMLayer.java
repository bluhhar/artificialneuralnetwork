package service;

import java.util.ArrayList;
import java.util.List;

public class LSTMLayer {
    private final List<LSTMCell> cells;
    private final int hiddenSize;

    public LSTMLayer(int inputSize, int hiddenSize) {
        this.hiddenSize = hiddenSize;
        this.cells = new ArrayList<>();
        for (int i = 0; i < hiddenSize; i++) {
            cells.add(new LSTMCell(inputSize));
        }
    }

    public double[] forward(double[][] sequence) {
        // sequence: [timeSteps][inputSize]
        double[] lastHiddenStates = new double[hiddenSize];

        for (double[] inputAtT : sequence) {
            for (int i = 0; i < hiddenSize; i++) {
                lastHiddenStates[i] = cells.get(i).forward(inputAtT)[0]; // берем только 1 значение (одномерно)
            }
        }

        return lastHiddenStates;
    }

    public List<LSTMCell> getCells() {
        return cells;
    }
}
