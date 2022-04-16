import * as tf from '@tensorflow/tfjs-node';

export const trainModel = async (model: tf.Sequential, inputs: tf.Tensor, labels: tf.Tensor) => {
    // Prepare the model for training.
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse'],
    });

    const batchSize = 32;
    const epochs = 1500;

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
    });
}
