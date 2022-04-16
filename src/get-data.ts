import fetch from 'node-fetch';

export type ApiCar = {
    Miles_per_Gallon: number;
    Horsepower: number;
}

export type Car = {
    mpg: number;
    horsepower: number;
}

export const getData = async (): Promise<Car[]> => {
    const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
    const carsData: ApiCar[] = await carsDataResponse.json() as ApiCar[];
    const cleaned = carsData.map(car => ({
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower,
    }))
        .filter(car => (car.mpg != null && car.horsepower != null));

    return cleaned;
}
