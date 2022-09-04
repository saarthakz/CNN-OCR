const fs = require("node:fs");
const tf = require("@tensorflow/tfjs-node");
const getImages = require("../util/getImages");

(async () => {

  // Creating the model
  const model = tf.sequential();
  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    activation: "relu",
    filters: 32,
    kernelSize: 3,
  }));
  model.add(tf.layers.maxPool2d({
    poolSize: 2,
  }));
  model.add(tf.layers.conv2d({
    activation: "relu",
    filters: 32,
    kernelSize: 3,
  }));
  model.add(tf.layers.maxPool2d({
    poolSize: 2,
  }));

  model.add(tf.layers.flatten());

  model.add(tf.layers.dense({
    activation: "relu",
    units: 140,
  }));
  model.add(tf.layers.dense({
    activation: "softmax",
    units: 10,
  }));

  model.compile({
    loss: "categoricalCrossentropy",
    optimizer: tf.train.sgd(0.02),
    metrics: ['accuracy']
  });

  model.summary();

  //Importing the data
  const trainingDataset = getImages("Training");
  const validationDataset = getImages("Validation");

  // Fitting the model
  await model.fitDataset(trainingDataset, {
    epochs: 5,
    validationData: validationDataset
  });

  await model.save("file://Models/CNN");

})();