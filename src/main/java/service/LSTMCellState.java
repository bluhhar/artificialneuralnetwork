package service;

public record LSTMCellState(
        double[] x_t,
        double[] h_prev,
        double[] c_prev,
        double[] h_t,
        double[] c_t,
        double[] inputGate,
        double[] forgetGate,
        double[] outputGate,
        double[] candidate) {}
