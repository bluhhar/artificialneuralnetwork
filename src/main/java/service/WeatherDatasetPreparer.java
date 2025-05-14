package service;

import entity.Weather;

import java.util.ArrayList;
import java.util.List;

public class WeatherDatasetPreparer {
    public static class DataPair {
        public float[][] inputs;
        public float[][] targets;

        public DataPair(float[][] inputs, float[][] targets) {
            this.inputs = inputs;
            this.targets = targets;
        }
    }

    public static DataPair prepareData(List<Weather> weatherList) {
        List<float[]> inputList = new ArrayList<>();
        List<float[]> targetList = new ArrayList<>();

        for (Weather w : weatherList) {
            if (w.getRainTomorrow() == null) continue;

            float[] input = new float[]{
                    w.getMinTemp(),
                    w.getMaxTemp(),
                    w.getRainfall(),
                    w.getEvaporation(),
                    w.getSunshine(),
                    w.getWindGustSpeed(),
                    w.getWindSpeed9am(),
                    w.getWindSpeed3pm(),
                    w.getHumidity9am(),
                    w.getHumidity3pm(),
                    w.getPressure9am(),
                    w.getPressure3pm(),
                    w.getCloud9am(),
                    w.getCloud3pm(),
                    w.getTemp9am(),
                    w.getTemp3pm()
            };

            boolean hasNaN = false;
            for (float v : input) {
                if (Float.isNaN(v)) {
                    hasNaN = true;
                    break;
                }
            }
            if (hasNaN) continue;

            float[] target = new float[]{
                    "Yes".equalsIgnoreCase(w.getRainTomorrow()) ? 1.0f : 0.0f
            };

            inputList.add(input);
            targetList.add(target);
        }

        float[][] inputs = inputList.toArray(new float[0][]);
        float[][] targets = targetList.toArray(new float[0][]);

        return new DataPair(inputs, targets);
    }
}
