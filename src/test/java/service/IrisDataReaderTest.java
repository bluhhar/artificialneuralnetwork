package service;

import entity.Iris;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import util.Fixtures;
import utility.IrisDataReader;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class IrisDataReaderTest {

    private Fixtures fixtures;

    @BeforeEach
    void setUp() {
        fixtures = new Fixtures();
    }

    @Test
    void testReadIris() {
        List<Iris> data = IrisDataReader.load(fixtures.getRecoursePath());
        assertEquals(150, data.size());
    }
}
