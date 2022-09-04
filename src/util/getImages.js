const fs = require("node:fs");
const tf = require("@tensorflow/tfjs-node");

function getImages(type) {
  const dataRoot = `MNIST/${type}-Data`;
  const labels = fs.readdirSync(dataRoot);

  const imageTensors = new Array();
  const imageLabels = new Array();

  labels.forEach((label) => {
    const files = fs.readdirSync(`${dataRoot}/${label}`);
    files.forEach((file) => {
      const img = fs.readFileSync(`${dataRoot}/${label}/${file}`);
      let imgTensor = tf.node.decodePng(img, 1);
      imgTensor = imgTensor.div(tf.scalar(255));
      imageTensors.push(imgTensor);
      imageLabels.push(label);
    });
  });

  const oneHots = tf.oneHot(tf.tensor1d(imageLabels, "int32"), 10).arraySync().map((oneHot) => tf.tensor1d(oneHot));
  const xDataset = tf.data.array(imageTensors);
  const yDataset = tf.data.array(oneHots);

  const xyDataset = tf.data
    .zip({ xs: xDataset, ys: yDataset })
    .shuffle(imageTensors.length)
    .batch(10);

  return xyDataset;

};

module.exports = getImages;