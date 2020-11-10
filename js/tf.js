// TODO: Binding sự kiện
document.addEventListener("DOMContentLoaded", run);

// TODO: Các hàm thực thi
// END---
// MAIN PROCESSING
async function run() {
  //Load and plot the original input data that we are goin to train on
  const data = await getData();
  const values = data.map((d) => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  tfvis.render.scatterplot(
    { name: "Horsepower v MPG" },
    { values },
    {
      xLabel: "Horsepower",
      yLabel: "MPG",
      height: 300,
    }
  );

  // Create the model
  // const model = createModel();

  //Upload model from local
  const model = await uploadModel();
  console.log(model);

  tfvis.show.modelSummary({ name: "Model Summary" }, model);

  // Convert the data to a form we can use for training
  const tensorData = convertToTensor(data);
  const { inputs, labels } = tensorData;

  // Train the model
  await trainModel(model, inputs, labels);
  console.log("Done Training. Model will be downloaded soon. Please wait!");
  alert("Done Training. Model will be downloaded soon. Please wait!");

  //Save model
  eximModel(model);
  

  // Make some predictions using the model and compare them to the original data
  testModel(model, data, cleanedTest, tensorData);
}

// TODO: Các hàm logic
// START---
// 1. Load, format and visualize the input data
var testingData = [];
var cleanedTest = [];
async function getData() {
  const carsDataResponse = await fetch(
    "https://storage.googleapis.com/tfjs-tutorials/carsData.json"
  );
  const carsData = await carsDataResponse.json();
  // console.log(carsData);
  // console.log(carsDataResponse);
  const trainingData = carsData.slice(0, 324);
  testingData = carsData.slice(324, 406);
  // console.log(trainingData);
  const cleaned = trainingData
    .map((car) => ({
      mpg: car.Miles_per_Gallon,
      horsepower: car.Horsepower,
    }))
    .filter((car) => car.mpg != null && car.horsepower != null);
  // console.log(cleaned);
  cleanedTest = testingData.map((car) => ({
    mpg: car.Miles_per_Gallon,
    horsepower: car.Horsepower,
  }));
  return cleaned;
}
// 2. Define the model architecture
// 2.1. Create model by hand
function createModel() {
  // Create a sequential model
  const model = tf.sequential();

  // Add a single input layer
  model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));

  //bz--I'll try to add some hidden layers

  model.add(tf.layers.dense({ units: 20, activation: "relu" }));
  //omg, when I add the above layer, it predicted so great. But how did it work?
  //Hmm, it's randomly true or false. The predicted result is not in the true path usually.
  // model.add(tf.layers.dense({units: 60, activation: 'linear'}));

  // Add an output layer
  model.add(tf.layers.dense({ units: 1, useBias: true }));

  return model;
}
// 2.2. Upload model
async function uploadModel() {
  const uploadJSONInput = document.getElementById("btnUploadJSON");
  const uploadWeightsInput = document.getElementById("btnUploadBin");
  const model = await tf.loadLayersModel(
    tf.io.browserFiles([uploadJSONInput.files[0], uploadWeightsInput.files[0]])
  );
  console.log("Finish build model");
  return model;
}
// 3. Prepare the data for training
function convertToTensor(data) {
  // Wrapping these calculations in a tidy will dispose any
  // intermediate tensors

  return tf.tidy(() => {
    // Step 1. Shuffle the data
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor
    const inputs = data.map((d) => d.horsepower);
    // alert("This is input \n" + inputs);
    const labels = data.map((d) => d.mpg);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]); // Convert to tensor .tensor2d(<array>,<shape_with_param_rowthencol-optional>,<dtype-optional>)
    // alert("This is inputTensor \n" + inputTensor);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    // Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor
      .sub(inputMin)
      .div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor
      .sub(labelMin)
      .div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      //Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    };
  });
}
// 4. Train the model
async function trainModel(model, inputs, labels) {
  // Prepare the model for training
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ["mse"],
  });

  const batchSize = 60;
  const epochs = 55;

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: "Training Performance" },
      ["loss", "mse"],
      { height: 200, callbacks: ["onEpochEnd"] }
    ),
  });
}
// 5. Download model
async function eximModel(model) {
  const saveResult = await document.getElementById("btnDownload").addEventListener("click", function(){
    model.save("downloads://model");
  });
  // const saveResult = await model.save("downloads://model");
  return saveResult;
}
// 6. Make predictions
function testModel(model, inputData, inputDataTesting, normalizationData) {
  const { inputMax, inputMin, labelMax, labelMin } = normalizationData;

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling we did earlier.
  const [xs, preds] = tf.tidy(() => {
    const xs = tf.linspace(0, 1, 100); //generate 100 examples
    const preds = model.predict(xs.reshape([100, 1]));

    const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);

    const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

    // Un-normalize the data
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] };
  });

  const originalPointsTraining = inputData.map((d) => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  const originalPointsTesting = inputDataTesting.map((d) => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  tfvis.render.scatterplot(
    { name: "Model Predictions vs Original Data (Training & Testing)" },
    {
      values: [originalPointsTraining, originalPointsTesting, predictedPoints],
      series: ["origTraining", "origTesting", "predicted"],
    },
    {
      xLabel: "Horsepower",
      yLabel: "MPG",
      heigth: 300,
    }
  );
}

//--------------Additional Functions----------------
// import * as tf from "@tensorflow/tfjs";
//Read file from upload
function readBlob(opt_startByte, opt_stopByte) {
  var files = document.getElementById("btnUploadJSON").files;
  if (!files.length) {
    alert("Please select a file!");
    return;
  }

  var file = files[0];
  var start = parseInt(opt_startByte) || 0;
  var stop = parseInt(opt_stopByte) || file.size - 1;

  var reader = new FileReader();

  // If we use onloadend, we need to check the readyState.
  reader.onloadend = function (evt) {
    if (evt.target.readyState == FileReader.DONE) {
      // DONE == 2
      // document.getElementById("byte_content").textContent = evt.target.result;
      let resultContent = document.getElementById("byte_content");
      resultContent.textContent = evt.target.result;
      // console.log(resultContent.textContent);
      // document.getElementById("byte_range").textContent = [
      //   "Read bytes: ",
      //   start + 1,
      //   " - ",
      //   stop + 1,
      //   " of ",
      //   file.size,
      //   " byte file",
      // ].join("");
    }
  };

  var blob = file.slice(start, stop + 1);
  reader.readAsBinaryString(blob);
}
document.querySelector(".readBytesButtons").addEventListener(
  "click",
  function (eventt) {
    if (eventt.target.tagName.toLowerCase() == "button") {
      var startByte = eventt.target.getAttribute("data-startbyte");
      var endByte = eventt.target.getAttribute("data-endbyte");
      readBlob(startByte, endByte);
    }
  },
  false
);
//end-read file from upload
