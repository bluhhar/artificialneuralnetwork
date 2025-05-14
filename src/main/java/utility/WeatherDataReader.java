package utility;

import entity.Weather;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class WeatherDataReader {
    public static List<Weather> loadWeatherData(String filePath) {
        List<Weather> weatherList = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            br.readLine(); // пропускаем заголовок

            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");

                if (values.length < 22) continue;

                Weather weather = new Weather();
                try {
                    weather.setDate(values[0]);
                    weather.setLocation(values[1]);
                    weather.setMinTemp(parseFloat(values[2]));
                    weather.setMaxTemp(parseFloat(values[3]));
                    weather.setRainfall(parseFloat(values[4]));
                    weather.setEvaporation(parseFloat(values[5]));
                    weather.setSunshine(parseFloat(values[6]));
                    weather.setWindGustSpeed(parseFloat(values[8]));
                    weather.setWindSpeed9am(parseFloat(values[10]));
                    weather.setWindSpeed3pm(parseFloat(values[11]));
                    weather.setHumidity9am(parseFloat(values[12]));
                    weather.setHumidity3pm(parseFloat(values[13]));
                    weather.setPressure9am(parseFloat(values[14]));
                    weather.setPressure3pm(parseFloat(values[15]));
                    weather.setCloud9am(parseFloat(values[16]));
                    weather.setCloud3pm(parseFloat(values[17]));
                    weather.setTemp9am(parseFloat(values[18]));
                    weather.setTemp3pm(parseFloat(values[19]));
                    weather.setRainToday(values[20]);
                    weather.setRainTomorrow(values[21]);
                } catch (NumberFormatException e) {
                    continue;
                }

                weatherList.add(weather);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return weatherList;
    }

    public static void normalizeWeatherData(List<Weather> data) {
        // Словари min/max значений
        Map<String, Float> minValues = new HashMap<>();
        Map<String, Float> maxValues = new HashMap<>();

        // Списки значений по полям
        Map<String, List<Float>> fieldValues = new HashMap<>();

        String[] fields = {
                "minTemp", "maxTemp", "rainfall", "evaporation", "sunshine",
                "windGustSpeed", "windSpeed9am", "windSpeed3pm",
                "humidity9am", "humidity3pm",
                "pressure9am", "pressure3pm",
                "cloud9am", "cloud3pm",
                "temp9am", "temp3pm"
        };

        // Собираем все значения по полям
        for (String field : fields) {
            List<Float> values = new ArrayList<>();
            for (Weather w : data) {
                Float value = getField(w, field);
                if (!Float.isNaN(value)) {
                    values.add(value);
                }
            }

            if (values.isEmpty()) {
                System.out.println("Поле пропущено из-за отсутствия данных: " + field);
                continue;
            }

            float min = Collections.min(values);
            float max = Collections.max(values);
            minValues.put(field, min);
            maxValues.put(field, max);
            fieldValues.put(field, values);
        }

        // Нормализация
        for (Weather w : data) {
            for (String field : fields) {
                float value = getField(w, field);
                if (!Float.isNaN(value)) {
                    float min = minValues.get(field);
                    float max = maxValues.get(field);
                    float norm = (value - min) / (max - min);
                    setField(w, field, norm);
                }
            }
        }
    }

    private static float parseFloat(String value) {
        try {
            return value == null || value.isEmpty() ? Float.NaN : Float.parseFloat(value);
        } catch (Exception e) {
            return Float.NaN;
        }
    }

    // Вспомогательные методы доступа по имени поля
    private static float getField(Weather w, String field) {
        return switch (field) {
            case "minTemp" -> w.getMinTemp();
            case "maxTemp" -> w.getMaxTemp();
            case "rainfall" -> w.getRainfall();
            case "evaporation" -> w.getEvaporation();
            case "sunshine" -> w.getSunshine();
            case "windGustSpeed" -> w.getWindGustSpeed();
            case "windSpeed9am" -> w.getWindSpeed9am();
            case "windSpeed3pm" -> w.getWindSpeed3pm();
            case "humidity9am" -> w.getHumidity9am();
            case "humidity3pm" -> w.getHumidity3pm();
            case "pressure9am" -> w.getPressure9am();
            case "pressure3pm" -> w.getPressure3pm();
            case "cloud9am" -> w.getCloud9am();
            case "cloud3pm" -> w.getCloud3pm();
            case "temp9am" -> w.getTemp9am();
            case "temp3pm" -> w.getTemp3pm();
            default -> Float.NaN;
        };
    }

    private static void setField(Weather w, String field, float value) {
        switch (field) {
            case "minTemp" -> w.setMinTemp(value);
            case "maxTemp" -> w.setMaxTemp(value);
            case "rainfall" -> w.setRainfall(value);
            case "evaporation" -> w.setEvaporation(value);
            case "sunshine" -> w.setSunshine(value);
            case "windGustSpeed" -> w.setWindGustSpeed(value);
            case "windSpeed9am" -> w.setWindSpeed9am(value);
            case "windSpeed3pm" -> w.setWindSpeed3pm(value);
            case "humidity9am" -> w.setHumidity9am(value);
            case "humidity3pm" -> w.setHumidity3pm(value);
            case "pressure9am" -> w.setPressure9am(value);
            case "pressure3pm" -> w.setPressure3pm(value);
            case "cloud9am" -> w.setCloud9am(value);
            case "cloud3pm" -> w.setCloud3pm(value);
            case "temp9am" -> w.setTemp9am(value);
            case "temp3pm" -> w.setTemp3pm(value);
        }
    }
}
