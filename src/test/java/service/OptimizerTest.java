package service;

import entity.Iris;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import service.activationfunction.impl.Sigmoid;
import service.optimizer.impl.AMSGradOptimizer;
import service.optimizer.impl.RmspropGravesOptimizer;
import service.optimizer.impl.SGDOptimizer;
import util.Exec;
import util.Fixtures;
import util.Timer;
import utility.IrisDataReader;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

@Slf4j
public class OptimizerTest {

    private Fixtures fixtures;
    private Timer timer;
    private List<Iris> dataset;

    @BeforeEach
    void setUp() {
        fixtures = new Fixtures();
        dataset = IrisDataReader.load(fixtures.getRecoursePath());
    }

    @Test
    void testSGDOptimizer() {
        String label = "Test SGDOptimizer";
        Exec exec = new Exec();
        timer = new Timer(label);
        boolean result = exec.test(
                dataset,
                new SGDOptimizer(),
                new Sigmoid(),
                0.01,
                1000,
                label,
                false);
        timer.stop();
        assertTrue(result);
    }

    @Test
    void testAMSGradOptimizer() {
        String label = "Test AMSGradOptimizer";
        Exec exec = new Exec();
        timer = new Timer(label);
        boolean result = exec.test(
                dataset,
                new AMSGradOptimizer(0.9, 0.999),
                new Sigmoid(),
                0.01,
                1000,
                label,
                true);
        timer.stop();
        assertTrue(result);
    }

    @Test
    void testRmspropGravesOptimizer() {
        String label = "Test RmspropGravesOptimizer";
        Exec exec = new Exec();
        timer = new Timer(label);
        boolean result = exec.test(
                dataset,
                new RmspropGravesOptimizer(0.95),
                new Sigmoid(),
                0.01,
                1000,
                label,
                true);
        timer.stop();
        assertTrue(result);
    }
}
