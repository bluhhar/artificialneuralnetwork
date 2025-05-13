package utility;

import java.util.Arrays;

public class KnowledgeDistiller {
    private double temperature = 2.0;
    private double alpha = 0.5;

    public double getTemperature() {
        return temperature;
    }

    public void setTemperature(double temperature) {
        this.temperature = temperature;
    }

    public double getAlpha() {
        return alpha;
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    // Расчет градиента
    public double[] calculateGradient(double[] studentOutput, double[] teacherOutput, double[] hardTargets) {
        if (studentOutput.length != teacherOutput.length || studentOutput.length != hardTargets.length) {
            throw new IllegalArgumentException("Arrays length mismatch");
        }

        double[] softTeacher = softmaxWithTemperature(teacherOutput, temperature);
        double[] softStudent = softmaxWithTemperature(studentOutput, temperature);
        double[] hardStudent = softmaxWithTemperature(studentOutput, 1.0); // Без температуры

        double[] gradient = new double[studentOutput.length];
        for (int i = 0; i < gradient.length; i++) {
            double softPart = (1 - alpha) * (softStudent[i] - softTeacher[i]);
            double hardPart = alpha * (hardStudent[i] - hardTargets[i]);
            gradient[i] = softPart + hardPart;
        }

        return gradient;
    }

    // Температурный softmax
    private double[] softmaxWithTemperature(double[] logits, double temp) {
        double max = Arrays.stream(logits).max().orElse(0.0);
        double[] expValues = new double[logits.length];
        double sum = 0.0;

        for (int i = 0; i < logits.length; i++) {
            expValues[i] = Math.exp((logits[i] - max) / temp);
            sum += expValues[i];
        }

        for (int i = 0; i < expValues.length; i++) {
            expValues[i] /= sum;
        }

        return expValues;
    }

    // Расчет потерь по Kullback-Leibler divergence
    public double calculateKD_Loss(double[] teacherOutput, double[] studentOutput, double T) {
        double[] softTeacher = softmaxWithTemperature(teacherOutput, T);
        double[] softStudent = softmaxWithTemperature(studentOutput, T);

        double loss = 0.0;
        for (int i = 0; i < softTeacher.length; i++) {
            loss += -softTeacher[i] * Math.log(Math.max(softStudent[i], 1e-10));
        }
        return loss;
    }
}

