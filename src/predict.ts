import * as tf from '@tensorflow/tfjs-node';
import {TrainData} from "./get-data";

export const predict = (model:tf.Sequential, input: Omit<TrainData, "exited">, params: any) => {
    const inputTensor = tf.tensor([input.Score, input.Tenure, input.Balance, input.Products, input.Salary], [1, 5, 1]);
    const normalizedInputs = inputTensor
        .sub(params.inputMin)
        .div(params.inputMax.sub(params.inputMin));

    const tensorResult = model.predict(normalizedInputs)
    const normalized = (tensorResult as any)
        .mul(params.labelMax.sub(params.labelMin))
        .add(params.labelMin);
    const result = normalized.dataSync();

    return result[0];
}
