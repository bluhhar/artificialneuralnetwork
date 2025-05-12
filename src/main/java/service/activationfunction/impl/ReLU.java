package service.activationfunction.impl;

import service.activationfunction.ActivationFunction;

public class ReLU implements ActivationFunction {

    @Override
    public double activate(double x) {
        return Math.max(0, x);
    }

    @Override
    public double[] activate(double[] x) {
        return new double[0];
    }

    @Override
    public double derivative(double x) {
        return x > 0 ? 1.0 : 0.0;
    }
}
