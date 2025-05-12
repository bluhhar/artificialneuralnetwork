package service.activationfunction.impl;

import service.activationfunction.ActivationFunction;

public class Tanh implements ActivationFunction {

    @Override
    public double activate(double x) {
        return Math.tanh(x);
    }

    @Override
    public double[] activate(double[] x) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = activate(x[i]);
        }
        return out;
    }

    @Override
    public double derivative(double x) {
        double y = activate(x);
        return 1 - y * y;
    }
}
