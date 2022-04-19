import * as tf from '@tensorflow/tfjs-node';

const createModel = (): tf.Sequential => {
    // Create a sequential model
    const model = tf.sequential();

    // Add a single input layer
    model.add(tf.layers.dense({
        inputShape: [2,1],
        units: 32,
        useBias: true,
        // weights: [
        //     // tf.tensor([1, 1], [1,2]),
        //     // tf.tensor([1,0.12], [2]),
        //     tf.randomUniform([2, 2], 1, 1),
        // ],
        activation:'relu',
    }));
    model.add(tf.layers.dense({units: 64, activation: 'relu'}));

    // model.add(tf.layers.reshape({
    //     targetShape: [1,2]
    // }))
    // model.add(tf.layers.flatten());

    // Add an output layer
    model.add(tf.layers.dense({units: 2, useBias: true, activation: 'softmax'}));

    return model;
}


export const createOrReadModel = async (isCreate: boolean, modelName: string): Promise<tf.Sequential> => {
    let model;
    if (isCreate) {
        model = createModel();
    } else {
        model = await tf.loadLayersModel(    `${modelName}/model.json`) as tf.Sequential;
    }

    return model;
}
