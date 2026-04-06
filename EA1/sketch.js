let classifier;
let img;

let label = "";
let confidence = "";

let canvasText = 'Drag an image file onto the canvas.';

function preload() {
    classifier = ml5.imageClassifier("MobileNet");
    img = loadImage("images/bird.jpg");
}

function setup() {
    createCanvas(400, 400);
    classifier.classify(img, gotResult);
    image(img, 300, 300);


    let dropArea = createCanvas(710, 400);
    dropArea.drop(gotFile);
    noLoop();
}

function draw() {
    background(220);

fill(255);
  noStroke();
  textSize(24);
  textAlign(CENTER);
  text(canvasText, width / 2, height / 2);

  describe(`Grey canvas with the text "${canvasText}" in the center.`);

/*
    myDiv = document.getElementById('myDiv');
    //console.log("This is my Div:" + myDiv);
    Plotly.newPlot(myDiv, [{
        x: [1, 2, 3, 4, 5],
        y: [1, 2, 4, 8, 16]
    }], {
        margin: { t: 0 }
    });
    */
}

function gotResult(results) {
    console.log(results);
}

function gotFile(file) {
  // If the file dropped into the canvas is an image,
  // create a variable called img to contain the image.
  // Remove this image file from the DOM and only
  // draw the image within the canvas.
  if (file.type === 'image') {
    // Pass in an empty string for the alt text. This should only be done with
    // decorative photos.
    let img = createImg(file.data, '').hide();
    image(img, 0, 0, width, height);
  } else {
    // If the file dropped into the canvas is not an image,
    // change the instructions to 'Not an image file!'
    canvasText = 'Not an image file!';
    redraw();
  }
}

//Plotly.newPlot('myDiv', data, layout);