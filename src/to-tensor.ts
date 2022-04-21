import * as tf from '@tensorflow/tfjs-node';
import {TrainData} from "./get-data";
import {toCategorical} from "./to-categorical";
import {normalize} from "./normalize";

const numberData:string[] = [
    "Score",
    "Tenure",
    "Balance",
    "Salary",
    "Age",
    "Products",
    "Card",
    "Active",
]

export const toTensors = (data: TrainData[], categoricalFeatures: Set<keyof TrainData>, testSize: number) => {
    const categoricalData: Record<string, any> = {};

    // non numbered params
    categoricalFeatures.forEach((f) => {
        categoricalData[f] = toCategorical(data, f);
    });

    const features: string[]= numberData.concat(Array.from(categoricalFeatures));

    // console.log(categoricalFeatures)
    const X: number[][] = data.map((r, i) =>
        features.flatMap((f) => {
            if (categoricalFeatures.has(f as keyof TrainData)) {
                return categoricalData[f][i] as number;
            }
            return r[f as keyof TrainData] as number;
        })
    );

    // console.log(X)
    const Y: number[] = data.map(item => item.Exited);
    const {tensor: X_t} = normalize(tf.tensor2d(X));
    const y = tf.tensor2d(toCategorical(data, "Exited"))
console.log(X_t.print())
//     console.log(y)
    y.print()
    const splitIdx = parseInt(String((1 - testSize) * data.length), 10);

    const [xTrain, xTest] = tf.split(X_t, [splitIdx, data.length - splitIdx]);
    const [yTrain, yTest] = tf.split(y, [splitIdx, data.length - splitIdx]);

    return [xTrain, xTest, yTrain, yTest];
};
