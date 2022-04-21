import * as tf from '@tensorflow/tfjs-node';

export const normalize = (tensor: tf.Tensor): {
    tensor: tf.Tensor,
    max: tf.Tensor<tf.Rank>,
    min: tf.Tensor<tf.Rank>,
} =>
{
    const max = tensor.max();
    const min = tensor.min();
    return {
        tensor: tf.div(
            tf.sub(tensor, min),
            tf.sub(max, min)
        ),
        max,
        min
    }

}
