import * as tf from '@tensorflow/tfjs-node';

const createModel = (): tf.Sequential => {
    // Create a sequential model
    const model = tf.sequential();

    // Add a single input layer
    model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
    model.add(tf.layers.dense({units: 50, activation: 'sigmoid'}));


    // Add an output layer
    model.add(tf.layers.dense({units: 1, useBias: true}));

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
