import * as tf from '@tensorflow/tfjs-node';
import {TrainData} from "./get-data";
import {toCategorical} from "./to-categorical";
import {normalize} from "./normalize";

export const toTensors = (data: TrainData[], categoricalFeatures: Set<keyof TrainData>, testSize: number) => {
    const categoricalData: Record<string, number[][]> = {};

    // non numbered params
    categoricalFeatures.forEach((f) => {
        categoricalData[f] = toCategorical(data, f);
    });

    const features: string[]= [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ].concat(Array.from(categoricalFeatures));

    // console.log(categoricalFeatures)
    const X: number[][] = data.map((r, i) =>
        features.flatMap((f) => {
            if (categoricalFeatures.has(f as keyof TrainData)) {
                return categoricalData[f][i] as number[];
            }
            return r[f as keyof TrainData] as number;
        })
    );

    // console.log(X[0])
    const {tensor: X_t} = normalize(tf.tensor2d(X));
    const y = tf.tensor(toCategorical(data, "Churn" as any));

    const splitIdx = parseInt(String((1 - testSize) * data.length), 10);

    const [xTrain, xTest] = tf.split(X_t, [splitIdx, data.length - splitIdx]);
    const [yTrain, yTest] = tf.split(y, [splitIdx, data.length - splitIdx]);

    return [xTrain, xTest, yTrain, yTest];
};
