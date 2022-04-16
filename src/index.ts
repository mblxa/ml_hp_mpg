import {getData} from "./get-data";
import {createOrReadModel} from "./create-or-read-model";
import {convertToTensor} from "./convert-to-tensor";
import {trainModel} from "./train-model";
import {predict} from "./predict";

(async () => {
    const modelName = 'model3';
    const modelPath = `file://./model/${modelName}`;
    const isNew = true;

    const result = await getData()
    const model = await createOrReadModel(isNew, modelPath);
    const data = convertToTensor(result)

    if (isNew) {
        const modelTrainingResult = await trainModel(model, data.inputs, data.labels);
        console.log('trained')
        await model.save(modelPath)
    }

    // console.log(predict(model, 10, data))
    // console.log(predict(model, 100, data))

})()
