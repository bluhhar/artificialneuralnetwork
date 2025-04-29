package entity;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@AllArgsConstructor
@Getter
@Setter
public class Iris {

    public double[] features; //длина и ширина чашелистика и лепестка

    public double[] label;    //кодировка класса

}
