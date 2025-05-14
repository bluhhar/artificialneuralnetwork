package service;

import entity.Iris;
import entity.Weather;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import util.Fixtures;
import util.Timer;
import utility.IrisDataReader;
import utility.WeatherDataReader;

import java.util.List;

public class ElmanRNNTest {

    private Fixtures fixtures;
    private Timer timer;
    private List<Iris> dataset;

    @BeforeEach
    void setUp() {
        fixtures = new Fixtures();
        dataset = IrisDataReader.load(fixtures.getRecoursePath());
    }

    @Test
    void testElmanRNN() {
        List<Weather> weatherData = WeatherDataReader.loadWeatherData(fixtures.getWeatherPath());
        WeatherDataReader.normalizeWeatherData(weatherData);

        WeatherDatasetPreparer.DataPair dataset = WeatherDatasetPreparer.prepareData(weatherData);

        float[][] inputs = dataset.inputs;
        float[][] targets = dataset.targets;

        ElmanRNN rnn = new ElmanRNN(16, 10, 1);

        // Тренировка
        rnn.train(inputs, targets, 100, 0.01f);

        // Оценка
        rnn.resetContext(); // сбросить состояние перед тестом
        rnn.evaluate(inputs, targets);
    }
}
