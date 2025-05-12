package service;

import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public class LSTMCellState {
    private final double[] x_t;
    private final double[] h_prev;
    private final double[] c_prev;
    private final double[] h_t;
    private final double[] c_t;

    private final double[] inputGate;
    private final double[] forgetGate;
    private final double[] outputGate;
    private final double[] candidate;
}
