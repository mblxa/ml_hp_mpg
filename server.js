const express = require('express')
const fs = require('fs');
const path = require('path');
const cors = require('cors');
const app = express();
const port = 3000;


const jsonData = require("./src/csvjson.ts")
const fileOptions = {
    root: path.join(__dirname)
};

app.use(cors())
app.get('/', (req, res) => {
    res.send('Hello World!')
})

const model = 'model4';
app.get('/model',async (req, res) => {
    // const file = fs.readFileSync('./model/model2/model.json', )
    res.sendFile(`./model/${model}/model.json`, fileOptions)
})
app.get('/weights.bin',async (req, res) => {
    // const file = fs.readFileSync('./model/model2/model.json', )
    res.sendFile(`./model/${model}/weights.bin`, fileOptions)
})

app.get('/data',async (req, res) => {
    res.json(jsonData)
})

app.listen(port, () => {
    console.log(`Example app listening on port ${port}`)
})
