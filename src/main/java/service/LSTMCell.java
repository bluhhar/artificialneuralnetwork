package service;

public class LSTMCell {
    private double[] weightsInputGate, weightsForgetGate, weightsOutputGate, weightsCell;
    private double[] hiddenState, cellState;
    private double biasInput, biasForget, biasOutput, biasCell;
    private int inputSize;

    public LSTMCell(int inputSize) {
        this.inputSize = inputSize;
        this.weightsInputGate = randomArray(inputSize);
        this.weightsForgetGate = randomArray(inputSize);
        this.weightsOutputGate = randomArray(inputSize);
        this.weightsCell = randomArray(inputSize);
        this.hiddenState = new double[inputSize];
        this.cellState = new double[inputSize];

        this.biasInput = Math.random() - 0.5;
        this.biasForget = Math.random() - 0.5;
        this.biasOutput = Math.random() - 0.5;
        this.biasCell = Math.random() - 0.5;
    }

    private double[] randomArray(int size) {
        double[] arr = new double[size];
        for (int i = 0; i < size; i++) arr[i] = Math.random() - 0.5;
        return arr;
    }

    public double[] forward(double[] input) {
        double[] inputGate = sigmoid(dot(input, weightsInputGate) + biasInput);
        double[] forgetGate = sigmoid(dot(input, weightsForgetGate) + biasForget);
        double[] outputGate = sigmoid(dot(input, weightsOutputGate) + biasOutput);
        double[] candidateCell = tanh(dot(input, weightsCell) + biasCell);

        for (int i = 0; i < inputSize; i++) {
            cellState[i] = forgetGate[i] * cellState[i] + inputGate[i] * candidateCell[i];
            hiddenState[i] = outputGate[i] * Math.tanh(cellState[i]);
        }

        return hiddenState;
    }

    private double dot(double[] x, double[] w) {
        double sum = 0;
        for (int i = 0; i < x.length; i++) sum += x[i] * w[i];
        return sum;
    }

    private double[] sigmoid(double x) {
        return new double[]{1.0 / (1.0 + Math.exp(-x))};
    }

    private double[] tanh(double x) {
        return new double[]{Math.tanh(x)};
    }
}
