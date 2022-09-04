const tf = require("@tensorflow/tfjs-node");
const fs = require("node:fs");

(async () => {
  const CNN = await tf.loadLayersModel("file://Models/CNN/model.json");
  const num = 6;
  const testFiles = fs.readdirSync(`./MNIST/Testing-Data/${num}`);

  const imageTensors = new Array();

  testFiles.forEach((testFile) => {
    //Image Data
    const imageData = fs.readFileSync(`./MNIST/Testing-Data/${num}/${testFile}`);

    //Tensor
    let imgTensor = tf.node.decodeJpeg(imageData, 1);

    //Normalized Tensor
    imgTensor = imgTensor.div(tf.scalar(255));
    imageTensors.push(imgTensor);
  });

  const xs = tf.stack(imageTensors);
  const CNN_predictions = CNN.predict(xs).arraySync();
  const results = CNN_predictions.map((prediction) => {
    const largest = Math.max(...prediction);
    return prediction.findIndex((elem) => elem == largest);
  });

  const numMap = {};
  results.forEach((result) => numMap[result] == undefined ? numMap[result] = 1 : numMap[result]++);
  console.log(numMap);
})();