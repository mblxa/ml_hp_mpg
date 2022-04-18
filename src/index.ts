import {getData, TrainData} from "./get-data";
import {createOrReadModel} from "./create-or-read-model";
import {convertToTensor} from "./convert-to-tensor";
import {trainModel} from "./train-model";
import {predict} from "./predict";

const predictData = (model: any, data: any) => {
    console.log(predict(model, 10, 3000, data))
    console.log(predict(model, 15, 3000, data))
    console.log(predict(model, 20, 3000, data))
    console.log();
    console.log(predict(model, 10, 3400, data))
    console.log(predict(model, 15, 3400, data))
    console.log(predict(model, 20, 3400, data))
    console.log();
    console.log(predict(model, 10, 4000, data))
    console.log(predict(model, 15, 4000, data))
    console.log(predict(model, 20, 4000, data))
    console.log();

    console.log(predict(model, 23, 2100, data))
    console.log(predict(model, 11, 2300, data))
    console.log(predict(model, 18, 2600, data))
}

const testModel = (model: any, testData: TrainData[], data: any) => {
    const tableData = [["weight", "acc", 'hp', 'pred', 'dif', '%'], ...testData.map(item => {
        const predicted = predict(model, item.acceleration, item.weight, data);
        return [
            item.weight,
            item.acceleration,
            item.horsepower,
            predicted,
            predicted - item.horsepower,
            parseFloat(((item.horsepower - predicted) / (item.horsepower / 100)).toFixed(2))
        ]
    })]

    const totalDif: number = tableData.slice(1).map(item => item[5]).reduce<number>((acc, item) => {
        acc += parseFloat(item)
        return acc
    }, 0)

    console.table(tableData)
    console.log(totalDif / testData.length)
}

(async () => {
    const modelName = 'model5';
    const modelPath = `file://./model/${modelName}`;
    const isNew = false;

    const result = await getData()
    const model = await createOrReadModel(isNew, modelPath);

    const coef = Math.ceil(result.length * 0.1);
    const trainData = result.slice(0, result.length - coef)
    const testData = result.slice(result.length - coef, result.length)

    const tensorData = convertToTensor(trainData)

    if (isNew) {
        const modelTrainingResult = await trainModel(model, tensorData.inputs, tensorData.labels);
        console.log('trained')
        await model.save(modelPath)
    }

    testModel(model, testData, tensorData)

    // predictData(model, tensorData)
})()
