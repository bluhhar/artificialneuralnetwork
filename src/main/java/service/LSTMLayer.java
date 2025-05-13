package service;

public class LSTMLayer {

    private final LSTMCell cell;
    private double[] lastHiddenState;
    private double[] lastCellState;

    public LSTMLayer(int inputSize, int hiddenSize) {
        this.cell = new LSTMCell(inputSize, hiddenSize);
        this.lastHiddenState = new double[hiddenSize];
        this.lastCellState = new double[hiddenSize];
    }

    public double[] forward(double[][] sequence) {
        double[] h = lastHiddenState;
        double[] c = lastCellState;

        for (double[] x : sequence) {
            h = cell.forward(x, h, c);
            c = cell.getHistory().getLast().c_t(); // обновляем c из последнего шага
        }

        lastHiddenState = h;
        lastCellState = c;
        return h;
    }

    public void backward(double[] outputGradient, double learningRate) {
        cell.backward(outputGradient, learningRate);
    }
}
