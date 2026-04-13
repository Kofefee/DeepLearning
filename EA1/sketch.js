let classifier;
let img;

let label = "";
let confidence = "";

let canvasText = 'Drag an image file onto the canvas.';

function preload() {
    classifier = ml5.imageClassifier("MobileNet");
    img = loadImage("images/good1.jpg");
}

function setup() {
    createCanvas(400, 400);
    classifier.classify(img, gotResult);
    image(img, 300, 300);
    // Drag and Drop
    let dropArea = createCanvas(710, 400);
    dropArea.drop(gotFile);
    noLoop();
}

function draw() {
    background(220);
    //Drag and Drop
    fill(255);
    noStroke();
    textSize(24);
    textAlign(CENTER);
    text(canvasText, width / 2, height / 2);

    describe(`Grey canvas with the text "${canvasText}" in the center.`);
    //Plotly.newPlot('myDiv', data, layout);
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

    // Labels und Werte extrahieren
    let labels = results.map(r => r.label);
    let values = results.map(r => r.confidence);

    // Plotly Daten aktualisieren
    data = [{
        values: values,
        labels: labels,
        type: 'pie',
        textinfo: 'label+percent', // zeigt Label + Prozent
        hoverinfo: 'label+percent+value'
    }];

    // Chart neu zeichnen
    Plotly.newPlot('myDiv', data, layout);
}

function gotFile(file) {
    //Drag and Drop
    if (file.type === 'image') {
        let img = createImg(file.data, '').hide();
        image(img, 0, 0, width, height);
    } else {
        canvasText = 'Not an image file!';
        redraw();
    }
}
