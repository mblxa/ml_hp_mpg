import * as tf from '@tensorflow/tfjs-node';

const createModel = (): tf.Sequential => {
    // Create a sequential model
    const model = tf.sequential();

    // Add a single input layer
    model.add(tf.layers.dense({
        inputShape: [2, 1],
        units: 2,
        useBias: true,
        weights: [
            tf.tensor([50, 1], [1,2]),
            tf.tensor([1,0.12], [2]),
            // tf.randomUniform([2], 1, 1),
        ],
    }));
    model.add(tf.layers.dense({units: 50, activation: 'sigmoid'}));
    model.add(tf.layers.flatten());

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
