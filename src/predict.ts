import * as tf from '@tensorflow/tfjs-node';

export const predict = (model:tf.Sequential, hp: number, weight: number, params: any) => {
    const inputTensor = tf.tensor3d([hp, weight], [1, 2, 1]);
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
