package service;

import lombok.Getter;
import service.activationfunction.ActivationFunction;
import service.activationfunction.impl.Sigmoid;
import service.activationfunction.impl.Tanh;

import java.util.ArrayList;
import java.util.List;

@Getter
public class LSTMCell {

    private final int inputSize;
    private final int hiddenSize;

    private double[][] weightsInput, weightsForget, weightsOutput, weightsCandidate;
    private double[] biasInput, biasForget, biasOutput, biasCandidate;

    private final List<LSTMCellState> history = new ArrayList<>();

    public LSTMCell(int inputSize, int hiddenSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        this.weightsInput = randomMatrix(hiddenSize, inputSize);
        this.weightsForget = randomMatrix(hiddenSize, inputSize);
        this.weightsOutput = randomMatrix(hiddenSize, inputSize);
        this.weightsCandidate = randomMatrix(hiddenSize, inputSize);

        this.biasInput = randomArray(hiddenSize);
        this.biasForget = randomArray(hiddenSize);
        this.biasOutput = randomArray(hiddenSize);
        this.biasCandidate = randomArray(hiddenSize);
    }

    public double[] forward(double[] x_t, double[] h_prev, double[] c_prev) {
        ActivationFunction tanh = new Tanh();
        ActivationFunction sigmoid = new Sigmoid();

        int H = hiddenSize;

        double[] i_t = sigmoid.activate(add(dot(weightsInput, x_t), biasInput));
        double[] f_t = sigmoid.activate(add(dot(weightsForget, x_t), biasForget));
        double[] o_t = sigmoid.activate(add(dot(weightsOutput, x_t), biasOutput));
        double[] g_t = tanh.activate(add(dot(weightsCandidate, x_t), biasCandidate));

        double[] c_t = new double[H];
        for (int i = 0; i < H; i++) {
            c_t[i] = f_t[i] * c_prev[i] + i_t[i] * g_t[i];
        }

        double[] h_t = new double[H];
        for (int i = 0; i < H; i++) {
            h_t[i] = o_t[i] * tanh.activate(c_t[i]);
        }

        history.add(new LSTMCellState(x_t, h_prev, c_prev, h_t, c_t, i_t, f_t, o_t, g_t));

        return h_t;
    }

    public void backward(double[] dLoss_dLastHidden, double learningRate) {
        Tanh tanh = new Tanh();
        Sigmoid sigmoid = new Sigmoid();

        int T = history.size();
        int H = hiddenSize;
        int X = inputSize;

        double[] dNext_h = dLoss_dLastHidden;
        double[] dNext_c = new double[H];

        // Градиенты по весам
        double[][] dW_i = new double[H][X];
        double[][] dW_f = new double[H][X];
        double[][] dW_o = new double[H][X];
        double[][] dW_g = new double[H][X];

        double[] db_i = new double[H];
        double[] db_f = new double[H];
        double[] db_o = new double[H];
        double[] db_g = new double[H];

        for (int t = T - 1; t >= 0; t--) {
            var state = history.get(t);

            double[] x = state.getX_t();
            double[] h_prev = state.getH_prev();
            double[] c_prev = state.getC_prev();
            double[] c = state.getC_t();
            double[] h = state.getH_t();

            double[] i = state.getInputGate();
            double[] f = state.getForgetGate();
            double[] o = state.getOutputGate();
            double[] g = state.getCandidate();

            double[] tanh_c = new double[H];
            for (int i1 = 0; i1 < H; i1++) tanh_c[i1] = tanh.activate(c[i1]);

            double[] dO = new double[H];
            double[] dC = new double[H];
            for (int i1 = 0; i1 < H; i1++) {
                dO[i1] = dNext_h[i1] * tanh_c[i1] * o[i1] * (1 - o[i1]);
                dC[i1] = dNext_h[i1] * o[i1] * (1 - tanh_c[i1] * tanh_c[i1]) + dNext_c[i1];
            }

            double[] dF = new double[H];
            double[] dI = new double[H];
            double[] dG = new double[H];
            for (int i1 = 0; i1 < H; i1++) {
                dF[i1] = dC[i1] * c_prev[i1] * f[i1] * (1 - f[i1]);
                dI[i1] = dC[i1] * g[i1] * i[i1] * (1 - i[i1]);
                dG[i1] = dC[i1] * i[i1] * (1 - g[i1] * g[i1]);
            }

            for (int i1 = 0; i1 < H; i1++) {
                for (int j = 0; j < X; j++) {
                    dW_i[i1][j] += dI[i1] * x[j];
                    dW_f[i1][j] += dF[i1] * x[j];
                    dW_o[i1][j] += dO[i1] * x[j];
                    dW_g[i1][j] += dG[i1] * x[j];
                }
                db_i[i1] += dI[i1];
                db_f[i1] += dF[i1];
                db_o[i1] += dO[i1];
                db_g[i1] += dG[i1];
            }

            // dNext_c для следующего шага
            for (int i1 = 0; i1 < H; i1++) {
                dNext_c[i1] = dC[i1] * f[i1];
            }

            // Здесь можно также посчитать dX и dH_prev (для глубоких RNN), если нужно
            dNext_h = new double[H]; // обнуляем, т.к. это однопроходная RNN
        }

        // Обновление весов вручную
        updateWeights(weightsInput, dW_i, biasInput, db_i, learningRate);
        updateWeights(weightsForget, dW_f, biasForget, db_f, learningRate);
        updateWeights(weightsOutput, dW_o, biasOutput, db_o, learningRate);
        updateWeights(weightsCandidate, dW_g, biasCandidate, db_g, learningRate);

        history.clear(); // очистка истории после backward
    }

    private void updateWeights(double[][] weights, double[][] gradW,
                               double[] biases, double[] gradB, double lr) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] += lr * gradW[i][j];
            }
            biases[i] += lr * gradB[i];
        }
    }

    // --- Utilities ---
    private double[] randomArray(int size) {
        double[] arr = new double[size];
        for (int i = 0; i < size; i++) arr[i] = Math.random() - 0.5;
        return arr;
    }

    private double[][] randomMatrix(int rows, int cols) {
        double[][] mat = new double[rows][cols];
        for (int i = 0; i < rows; i++) mat[i] = randomArray(cols);
        return mat;
    }

    private double[] dot(double[][] weights, double[] x) {
        double[] out = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            out[i] = 0;
            for (int j = 0; j < x.length; j++) {
                out[i] += weights[i][j] * x[j];
            }
        }
        return out;
    }

    private double[] add(double[] a, double[] b) {
        double[] res = new double[a.length];
        for (int i = 0; i < a.length; i++) res[i] = a[i] + b[i];
        return res;
    }
}
