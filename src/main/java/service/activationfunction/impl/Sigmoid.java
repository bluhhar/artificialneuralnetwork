package service.activationfunction.impl;

import service.activationfunction.ActivationFunction;

public class Sigmoid implements ActivationFunction {

    @Override
    public double activate(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    @Override
    public double derivative(double x) {
        double sig = activate(x);
        return sig * (1 - sig);
    }
}
