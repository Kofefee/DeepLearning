let classifier;
let img;

let label = "";
let confidence = "";

function preload() {
  classifier = ml5.imageClassifier("MobileNet");
  img = loadImage("images/bird.png");
}

function setup() {
  createCanvas(400, 400);
  classifier.classify(img, gotResult);
  image(img, 300, 300);
}

function draw() {
  background(220);
}

function gotResult(results) {
  console.log(results);
  
    fill(255);
  stroke(0);
  textSize(18);
  label = "Label: " + results[0].label;
  confidence = "Confidence: " + nf(results[0].confidence, 0,2);
  text(label, 10, 360);
  text(confidence, 10, 380);
  console.log(label);
}

var data = [{
  values: [19, 26, 55],
  labels: ['Residential', 'Non-Residential', 'Utility'],
  type: 'pie'
}];

var layout = {
  height: 400,
  width: 500
};
myDiv = document.getElementById('myDiv');
console.log(myDiv);
Plotly.newPlot('myDiv', data, layout);