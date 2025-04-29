package service;

import entity.Iris;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import service.activationfunction.impl.Sigmoid;
import service.optimizer.impl.AMSGradOptimizer;
import service.optimizer.impl.RmspropGravesOptimizer;
import service.optimizer.impl.SGDOptimizer;
import service.regularizer.impl.GroupLassoRegularizer;
import util.Exec;
import util.Fixtures;
import util.Timer;
import utility.IrisDataReader;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

@Slf4j
public class RegularizerTest {

    private Fixtures fixtures;
    private Timer timer;
    private List<Iris> dataset;

    @BeforeEach
    void setUp() {
        fixtures = new Fixtures();
        dataset = IrisDataReader.load(fixtures.getRecoursePath());
    }

    @Test
    void testSGDRegularizer() {
        String label = "Test SGDRegularizer";
        Exec exec = new Exec();
        timer = new Timer(label);
        boolean result = exec.test(
                dataset,
                new SGDOptimizer(new GroupLassoRegularizer(0.001)),
                new Sigmoid(),
                0.01,
                1000,
                label,
                false);
        timer.stop();
        assertTrue(result);
    }

    @Test
    void testAMSGradRegularizer() {
        String label = "Test AMSGradRegularizer";
        Exec exec = new Exec();
        timer = new Timer(label);
        boolean result = exec.test(
                dataset,
                new AMSGradOptimizer(new GroupLassoRegularizer(0.001)),
                new Sigmoid(),
                0.01,
                1000,
                label,
                true);
        timer.stop();
        assertTrue(result);
    }

    @Test
    void testRmspropGravesRegularizer() {
        String label = "Test RmspropGravesRegularizer";
        Exec exec = new Exec();
        timer = new Timer(label);
        boolean result = exec.test(
                dataset,
                new RmspropGravesOptimizer(new GroupLassoRegularizer(0.001)),
                new Sigmoid(),
                0.01,
                1000,
                label,
                false);
        timer.stop();
        assertTrue(result);
    }
}
