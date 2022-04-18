import fetch from 'node-fetch';
import * as tf from "@tensorflow/tfjs-node";

export type ApiCar = {
    Acceleration: number;
    Cylinders: number;
    Weight_in_lbs: number;
    Miles_per_Gallon: number;
    Horsepower: number;
}

export type Car = {
    acceleration: number;
    cylinders: number;
    weight: number;
    mpg: number;
    horsepower: number;
}

export type TrainData = Pick<Car, "acceleration" | "horsepower" | "weight">

export const getData = async (): Promise<TrainData[]> => {
    const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
    const carsData: ApiCar[] = await carsDataResponse.json() as ApiCar[];
    const cleaned = carsData.map<TrainData>(car => ({
        acceleration: car.Acceleration,
        horsepower: car.Horsepower,
        weight: car.Weight_in_lbs,
    }))
        .filter(car => (car.acceleration != null && car.horsepower != null && car.weight != null));

    tf.util.shuffle(cleaned)
    return cleaned;
}
