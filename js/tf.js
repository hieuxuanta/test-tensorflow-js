// START---
// import * as tf from "@tensorflow/tfjs";

// 3. Load, format and visualize the input data
/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
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
  cleanedTest = testingData
    .map((car) => ({
      mpg: car.Miles_per_Gallon,
      horsepower: car.Horsepower,
    }));
  return cleaned;
}

// 4. Define the model architecture
function createModel() {
  // Create a sequential model
  const model = tf.sequential();

  // Add a single input layer
  model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));

  //bz--I'll try to add some hidden layers

  
  model.add(tf.layers.dense({units: 60, activation: 'relu'}));
  //omg, when I add the above layer, it predicted so great. But how did it work?
  //But, it's randomly true and false. The predicted result is not in the true path usually.
  

  // Add an output layer
  model.add(tf.layers.dense({ units: 1, useBias: true }));

  return model;
}

// 5. Prepare the data for training
/**
 * Convert the input data to tensors that we can use for 
 * machine learning.
 * Do practices of _shuffing_ the data and _normalizing_ the data
 * MPG on the y-axis
 */
function convertToTensor(data){
  // Wrapping these calculations in a tidy will dispose any
  // intermediate tensors

  return tf.tidy(() => {
    // Step 1. Shuffle the data
    tf.util.shuffle(data);
    
    // Step 2. Convert data to Tensor
    const inputs = data.map(d => d.horsepower);
    // alert("This is input \n" + inputs);
    const labels = data.map(d => d.mpg);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]); // Convert to tensor .tensor2d(<array>,<shape_with_param_rowthencol-optional>,<dtype-optional>)
    // alert("This is inputTensor \n" + inputTensor);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    // Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      //Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    }
  });
}


// 6. Train the model
async function trainModel(model, inputs, labels){
  // Prepare the model for training
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });

  const batchSize = 25;
  const epochs = 50;
  
  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      {name: 'Training Performance'},
      ['loss', 'mse'],
      {height: 200, callbacks: ['onEpochEnd']}
    )
  })
}


// 7. Make predictions
function testModel(model, inputData, inputDataTesting, normalizationData){
  const {inputMax, inputMin, labelMax, labelMin} = normalizationData;

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling we did earlier.
  const [xs, preds] = tf.tidy(() => {
    const xs = tf.linspace(0, 1, 100); //generate 100 examples
    const preds = model.predict(xs.reshape([100, 1]));

    const unNormXs = xs
      .mul(inputMax.sub(inputMin))
      .add(inputMin);
    
    const unNormPreds = preds
      .mul(labelMax.sub(labelMin))
      .add(labelMin);
    
    // Un-normalize the data
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    return {x: val, y: preds[i]}
  });

  const originalPointsTraining = inputData.map(d => ({
    x: d.horsepower, y: d.mpg,
  }));

  const originalPointsTesting = inputDataTesting.map(d => ({
    x: d.horsepower, y: d.mpg,
  }));

  tfvis.render.scatterplot(
    {name: 'Model Predictions vs Original Data (Training & Testing)'},
    {values: [originalPointsTraining, originalPointsTesting, predictedPoints], series: ['origTraining', 'origTesting', 'predicted']},
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      heigth: 300
    }
  );
}


// END---
// This part is to end the file. His target is on the point of running program (like a main() function), including plotting on the webpage (visualize)
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
  const model = createModel();
  tfvis.show.modelSummary({ name: "Model Summary" }, model);

  // Convert the data to a form we can use for training
  const tensorData = convertToTensor(data);
  const {inputs, labels} = tensorData;

  // Train the model
  await trainModel(model, inputs, labels);
  console.log('Done Training.....');

  // Make some predictions using the model and compare them to the original data
  testModel(model, data, cleanedTest, tensorData);
}

document.addEventListener("DOMContentLoaded", run);



//----------------
/*
// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

// Generate some synthetic data for training.
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// Train the model using the data.
model.fit(xs, ys, { epochs: 10 }).then(() => {
  // Use the model to do inference on a data point the model hasn't seen before:
  model.predict(tf.tensor2d([5], [1, 1])).print();
  // Open the browser devtools to see the output
});
*/