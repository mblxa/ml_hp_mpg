import * as tf from '@tensorflow/tfjs-node';
import {TrainData} from "./get-data";

export const convertToTensor = (data: TrainData[]) => {
    // Wrapping these calculations in a tidy will dispose any
    // intermediate tensors.

    return tf.tidy(() => {
        // Step 1. Shuffle the data
        tf.util.shuffle(data);

        // Step 2. Convert data to Tensor
        // const inputs = data.map(d => d.Score)
        const labels = data.map(d => d.Exited);

        const inputs: number[] = [];
        data.forEach((item) => {
            inputs.push(item.Score)
            inputs.push(item.Tenure)
            inputs.push(item.Balance)
            inputs.push(item.Products)
            inputs.push(item.Salary)
        })

        const inputTensor = tf.tensor2d([inputs], [2, 1]);
        // const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();

        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

        return {
            // inputs: inputTensor,
            // labels: labelTensor,
            inputs: normalizedInputs,
            labels: normalizedLabels,
            // Return the min/max bounds so we can use them later.
            inputMax,
            inputMin,
            labelMax,
            labelMin,
        }
    });
}
