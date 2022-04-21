import * as tf from '@tensorflow/tfjs-node';

export const oneHot = (val: any, categoryCount: number) =>
    Array.from(tf.oneHot(val, categoryCount).dataSync());
