const tf = require( "@tensorflow/tfjs-node");
const jsonData = require ("./src/example.js");

const prepareData = () => {
    return jsonData
};

// normalized = (value − min_value) / (max_value − min_value)
const normalize = (tensor) =>
    tf.div(
        tf.sub(tensor, tf.min(tensor)),
        tf.sub(tf.max(tensor), tf.min(tensor))
    );

const oneHot = (val, categoryCount) =>
    Array.from(tf.oneHot(val, categoryCount).dataSync());

const toCategorical = (data, column) => {
    const values = data.map((r) => r[column]);
    const uniqueValues = new Set(values);

    const mapping = {};

    Array.from(uniqueValues).forEach((i, v) => {
        mapping[i] = v;
    });

    const encoded = values
        .map((v) => {
            if (!v) {
                return 0;
            }
            return mapping[v];
        })
        .map((v) => oneHot(v, uniqueValues.size));

    return encoded;
};

const toTensors = (data, categoricalFeatures, testSize) => {
    const categoricalData = {};
    categoricalFeatures.forEach((f) => {
        categoricalData[f] = toCategorical(data, f);
    });

    const features = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ].concat(Array.from(categoricalFeatures));

    const X = data.map((r, i) =>
        features.flatMap((f) => {
            if (categoricalFeatures.has(f)) {
                return categoricalData[f][i];
            }

            return r[f];
        })
    );

    console.log(X)
    const X_t = normalize(tf.tensor2d(X));
    const y = tf.tensor(toCategorical(data, "Churn"));
    X_t.print();
    y.print();
// console.log(X_t)
// console.log(y)
    const splitIdx = parseInt((1 - testSize) * data.length, 10);

    const [xTrain, xTest] = tf.split(X_t, [splitIdx, data.length - splitIdx]);
    const [yTrain, yTest] = tf.split(y, [splitIdx, data.length - splitIdx]);

    return [xTrain, xTest, yTrain, yTest];
};

const trainModel = async (xTrain, yTrain) => {
    const model = tf.sequential();
    model.add(
        tf.layers.dense({
            units: 32,
            activation: "relu",
            inputShape: [xTrain.shape[1]]
        })
    );

    model.add(
        tf.layers.dense({
            units: 64,
            activation: "relu"
        })
    );

    model.add(tf.layers.dense({ units: 2, activation: "softmax" }));

    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: "binaryCrossentropy",
        metrics: ["accuracy"]
    });

    await model.fit(xTrain, yTrain, {
        batchSize: 64,
        epochs: 500,
        shuffle: true,
        validationSplit: 0.1,
    });

    return model;
};

const run = async () => {
    const data = prepareData();

    const categoricalFeatures = new Set([
        "TechSupport",
        "Contract",
        "PaymentMethod",
        "gender",
        "Partner",
        "InternetService",
        "Dependents",
        "PhoneService",
        "TechSupport",
        "StreamingTV",
        "PaperlessBilling"
    ]);

    const [xTrain, xTest, yTrain, yTest] = toTensors(
        data,
        categoricalFeatures,
        0.1
    );

    const model = await trainModel(xTrain, yTrain);

    const result = model.evaluate(xTest, yTest, {
        batchSize: 32
    });
    result[0].print();
    result[1].print();
};
(async () => {
    await run();
})()
