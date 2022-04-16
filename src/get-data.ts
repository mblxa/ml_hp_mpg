import fetch from 'node-fetch';

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

export type TrainData = Pick<Car, "acceleration" | "horsepower">

export const getData = async (): Promise<TrainData[]> => {
    const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
    const carsData: ApiCar[] = await carsDataResponse.json() as ApiCar[];
    const cleaned = carsData.map<TrainData>(car => ({
        acceleration: car.Acceleration,
        horsepower: car.Horsepower,
    }))
        .filter(car => (car.acceleration != null && car.horsepower != null));

    return cleaned;
}
