package util;

import lombok.extern.slf4j.Slf4j;

@Slf4j
public class Timer {

    private final long startTime;
    private final String label;

    public Timer(String label) {
        this.label = label;
        this.startTime = System.nanoTime();
    }

    public void stop() {
        long elapsedTime = (System.nanoTime() - startTime) / 1_000_000;
        log.info("{} elapsed time {} ms.", label, elapsedTime);
    }
}
