public class GroupLassoRegularizer implements Regularizer {
    private double lambda;

    public GroupLassoRegularizer(double lambda) {
        this.lambda = lambda;
    }

    @Override
    public double[] computeGradient(double[] weights) {
        double l2Norm = 0.0;
        for (double w : weights) {
            l2Norm += w * w;
        }
        l2Norm = Math.sqrt(l2Norm) + 1e-8; // чтобы не было деления на ноль

        double[] regGrad = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            regGrad[i] = lambda * weights[i] / l2Norm;
        }
        return regGrad;
    }
}
