package service;

import entity.Iris;
import entity.Weather;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import util.Fixtures;
import utility.IrisDataReader;
import utility.WeatherDataReader;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class WeatherDataReaderTest {

    private Fixtures fixtures;

    @BeforeEach
    void setUp() {
        fixtures = new Fixtures();
    }

    @Test
    void testReadIris() {
        List<Weather> weatherData = WeatherDataReader.loadWeatherData(fixtures.getWeatherPath());
        WeatherDataReader.normalizeWeatherData(weatherData);
        System.out.println("Пример нормализованных данных:");
        System.out.println(weatherData.getFirst());
    }
}
