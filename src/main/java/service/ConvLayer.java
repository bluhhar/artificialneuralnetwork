package service;

import lombok.Getter;
import utility.ActivationFunctions;
import utility.MatrixUtils;

import java.util.*;
import java.util.stream.Collectors;

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
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                kernel[i][j] = (rnd.nextDouble() * 2 - 1) * scale;
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
        int width = lastInput.length;
        int height = lastInput[0].length;
        double[][] inputGradient = new double[width][height];

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
                            if (inputX < width && inputY < height) {
                                inputGradient[inputX][inputY] += deriv * kernel[i][j];
                            }
                        }
                    }
                }
            }

            // Обновление весов ядра
            for (int i = 0; i < kernel.length; i++) {
                for (int j = 0; j < kernel[0].length; j++) {
                    double gradSum = 0;
                    for (int x = 0; x < grad.length; x++) {
                        for (int y = 0; y < grad[0].length; y++) {
                            int inputX = x * stride + i;
                            int inputY = y * stride + j;
                            if (inputX < width && inputY < height) {
                                gradSum += grad[x][y] * lastInput[inputX][inputY];
                            }
                        }
                    }
                    if (kernel[i][j] != 0) {
                        kernel[i][j] -= learningRate * gradSum;
                    }
                }
            }
        }

        return inputGradient;
    }

    public void pruneKernels(double sparsity) {
        if (sparsity <= 0 || sparsity >= 1)
            throw new IllegalArgumentException("Sparsity must be in (0, 1)");

        List<Double> allWeights = kernels.stream()
                .flatMap(k -> Arrays.stream(k)
                        .flatMapToDouble(Arrays::stream)
                        .boxed()
                        .filter(w -> w != 0)
                        .map(Math::abs))
                .collect(Collectors.toList());

        if (allWeights.isEmpty()) {
            System.out.println("Ошибка: Нет весов для прунинга.");
            return;
        }

        if (allWeights.stream().distinct().count() == 1) {
            System.out.println("Предупреждение: Все веса одинаковы, прунинг может быть бесполезен.");
        }

        Collections.sort(allWeights);
        int thresholdIndex = (int) (sparsity * (allWeights.size() - 1));
        double threshold = allWeights.get(Math.max(0, Math.min(thresholdIndex, allWeights.size() - 1)));

        int total = allWeights.size();
        int pruned = 0;

        for (double[][] kernel : kernels) {
            for (int i = 0; i < kernel.length; i++) {
                for (int j = 0; j < kernel[0].length; j++) {
                    if (Math.abs(kernel[i][j]) < threshold && kernel[i][j] != 0) {
                        kernel[i][j] = 0;
                        pruned++;
                    }
                }
            }
        }

        System.out.printf("Прунинг сверточного слоя: обнулено %d из %d весов (%.2f%%).\n",
                pruned, total, pruned * 100.0 / total);
    }

    public void quantizeKernels() {
        for (double[][] kernel : kernels) {
            List<Double> weights = Arrays.stream(kernel)
                    .flatMapToDouble(Arrays::stream)
                    .filter(w -> w != 0)
                    .boxed()
                    .collect(Collectors.toList());

            if (weights.isEmpty()) continue;

            double min = Collections.min(weights);
            double max = Collections.max(weights);
            double scale = (max - min) / 255.0;

            for (int i = 0; i < kernel.length; i++) {
                for (int j = 0; j < kernel[0].length; j++) {
                    if (kernel[i][j] != 0) {
                        int quantized = (int) ((kernel[i][j] - min) / scale);
                        kernel[i][j] = min + quantized * scale;
                    }
                }
            }
        }
        System.out.println("Квантизация Conv: завершена");
    }
}

