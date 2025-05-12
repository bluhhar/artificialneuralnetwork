package service.activationfunction;

public interface ActivationFunction {

    double activate(double x);

    double[] activate(double[] x);

    double derivative(double x);
}
