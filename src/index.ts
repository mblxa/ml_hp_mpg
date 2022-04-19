import {getData, TrainData} from "./get-data";
import {createOrReadModel} from "./create-or-read-model";
import {convertToTensor} from "./convert-to-tensor";
import {trainModel} from "./train-model";
import {predict} from "./predict";
import * as tf from "@tensorflow/tfjs-node";
import {Tensor} from "@tensorflow/tfjs-node";

const testModel = (model: any, testData: TrainData[], data: any) => {
    const tableData = [["score", "exited", 'pred', ], ...testData.map(item => {
        const predicted = predict(model, item, data);
        return [
            item.Score,
            item.Exited,
            predicted,
        ]
    })]

    console.table(tableData)
}

(async () => {
    const modelName = 'model5';
    const modelPath = `file://./model/${modelName}`;
    const isNew = true;

    const result = await getData()
    const model = await createOrReadModel(isNew, modelPath);

    const coef = Math.ceil(result.length * 0.1);
    const trainData = result.slice(0, result.length - coef)
    const testData = result.slice(result.length - coef, result.length)

    const tensorData = convertToTensor(trainData)

    if (isNew) {
        const modelTrainingResult = await trainModel(model, tensorData.inputs, tensorData.labels);
        console.log('trained')
        const evaluateResult = modelTrainingResult.evaluate(
            tf.tensor([testData[0].Score, testData[0].Tenure, testData[0].Balance, testData[0].Products, testData[0].Salary], [1, 5, 1]),
            tf.tensor(testData[0].Exited, [1]),
            {
                batchSize: 32
            }
        );

        (evaluateResult  as any)[0].print()
        (evaluateResult  as any)[1].print()
        await model.save(modelPath)
    }

    // evaluateResult[0].print();
    // evaluateResult[1].print();

    // testModel(model, testData, tensorData)
})()
