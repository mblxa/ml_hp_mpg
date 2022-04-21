import * as tf from '@tensorflow/tfjs-node';

export const createModel = (xTrain: tf.Tensor): tf.Sequential => {
    const model = tf.sequential();
    model.add(
        tf.layers.dense({
            units: 32,
            activation: "relu",
            inputShape: [xTrain.shape[1] as any]
        })
    );

    model.add(
        tf.layers.dense({
            units: 64,
            activation: "relu"
        })
    );

    model.add(tf.layers.dense({ units: 2, activation: "softmax" }));

    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: "binaryCrossentropy",
        metrics: ["accuracy"]
    });

    return  model
}

//
// export const createOrReadModel = async (isCreate: boolean, modelName: string): Promise<tf.Sequential> => {
//     let model;
//     if (isCreate) {
//         model = createModel();
//     } else {
//         model = await tf.loadLayersModel(    `${modelName}/model.json`) as tf.Sequential;
//     }
//
//     return model;
// }
