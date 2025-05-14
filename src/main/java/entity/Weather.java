package entity;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class Weather {
    private String date;
    private String location;
    private float minTemp;
    private float maxTemp;
    private float rainfall;
    private float evaporation;
    private float sunshine;
    private float windGustSpeed;
    private float windSpeed9am;
    private float windSpeed3pm;
    private float humidity9am;
    private float humidity3pm;
    private float pressure9am;
    private float pressure3pm;
    private float cloud9am;
    private float cloud3pm;
    private float temp9am;
    private float temp3pm;
    private String rainToday;
    private String rainTomorrow;
}

