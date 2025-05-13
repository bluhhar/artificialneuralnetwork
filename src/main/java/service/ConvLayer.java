package service;

import java.util.*;

import lombok.Getter;
import utility.MatrixUtils;
import utility.ActivationFunctions;

@Getter
public class ConvLayer {
    private final int kernelSize;
    private final int numKernels;
    private final int stride;
    private final double learningRate;

    private final List<double[][]> kernels;
    private List<double[][]> lastOutputs;
    private double[][] lastInput;

    private final Random rnd = new Random();

    public ConvLayer(int kernelSize, int numKernels, double learningRate, int stride) {
        this.kernelSize = kernelSize;
        this.numKernels = numKernels;
        this.learningRate = learningRate;
        this.stride = stride;

        this.kernels = new ArrayList<>();
        for (int i = 0; i < numKernels; i++) {
            this.kernels.add(initKernel(kernelSize));
        }
    }

    private double[][] initKernel(int size) {
        double[][] kernel = new double[size][size];
        double scale = Math.sqrt(2.0 / (size * size));
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                kernel[i][j] = (rnd.nextDouble() * 2 - 1) * scale;
            }
        }
        return kernel;
    }

    public List<double[][]> forward(double[][] input) {
        this.lastInput = input;
        this.lastOutputs = new ArrayList<>();

        for (double[][] kernel : kernels) {
            double[][] convolved = MatrixUtils.convolve(input, kernel, stride);
            double[][] activated = ActivationFunctions.relu(convolved, 1e-7);
            lastOutputs.add(activated);
        }

        return lastOutputs;
    }

    public double[][] backward(List<double[][]> gradient) {
        int inputWidth = lastInput.length;
        int inputHeight = lastInput[0].length;
        double[][] inputGradient = new double[inputWidth][inputHeight];

        for (int k = 0; k < kernels.size(); k++) {
            double[][] kernel = kernels.get(k);
            double[][] grad = gradient.get(k);
            double[][] output = lastOutputs.get(k);

            for (int x = 0; x < grad.length; x++) {
                for (int y = 0; y < grad[0].length; y++) {
                    double deriv = grad[x][y] * (output[x][y] > 0 ? 1 : 0);

                    for (int i = 0; i < kernel.length; i++) {
                        for (int j = 0; j < kernel[0].length; j++) {
                            int inputX = x * stride + i;
                            int inputY = y * stride + j;

                            if (inputX < inputWidth && inputY < inputHeight) {
                                inputGradient[inputX][inputY] += deriv * kernel[i][j];
                            }
                        }
                    }
                }
            }

            // Обновление ядра
            for (int i = 0; i < kernel.length; i++) {
                for (int j = 0; j < kernel[0].length; j++) {
                    double gradSum = 0;

                    for (int x = 0; x < grad.length; x++) {
                        for (int y = 0; y < grad[0].length; y++) {
                            int inputX = x * stride + i;
                            int inputY = y * stride + j;

                            if (inputX < inputWidth && inputY < inputHeight) {
                                gradSum += grad[x][y] * lastInput[inputX][inputY];
                            }
                        }
                    }

                    kernel[i][j] -= learningRate * gradSum;
                }
            }
        }

        return inputGradient;
    }
}
