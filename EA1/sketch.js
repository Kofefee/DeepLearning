let classifier;
let img;

let label = "";
let confidence = "";

let canvasText = 'Drag an image file onto the canvas.';

var layout = {
    height: 400,
    width: 500
};

let defaultImages = [
    "images/good1.jpg",
    "images/good2.jpg",
    "images/good3.jpg",
    "images/bad1.jpg",
    "images/bad2.jpg",
    "images/bad3.jpg"
];

function preload() {
    classifier = ml5.imageClassifier("MobileNet");
    img = loadImage("images/good1.jpg");

    loadDefaultImages(); // HIER aufrufen!
}

function setup() {
    //createCanvas(400, 400);

    //classifier.classify(img, gotResult);
    //image(img, 300, 300);
    // Drag and Drop
    let dropArea = createCanvas(710, 400);
    dropArea.drop(gotFile);
    noLoop();


    // Default-Bilder laden
    //loadDefaultImages();
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
    let labels = results.map(r => r.label.substring(0, 30));
    let values = results.map(r => r.confidence);

    // Plotly Daten aktualisieren
    data = [{
        values: values,
        labels: labels,
        type: 'pie',
        //textinfo: 'label+percent', // zeigt Label + Prozent
        hoverinfo: 'label+percent+value'
    }];

    // Chart neu zeichnen
    Plotly.newPlot('myDiv', data, layout);
}

function gotFile(file) {
    if (file.type === 'image') {
        let imgElement = createImg(file.data, '').hide();

        // Klassifizieren und Bild + Chart erzeugen
        classifier.classify(imgElement.elt, (results) => {
            createResultRow(file.data, results, 'uploadedImages');
        });

    } else {
        canvasText = 'Not an image file!';
        redraw();
    }
}

function createResultRow(imageSrc, results, divName) {
    let container = document.getElementById(divName);

    // Row erstellen
    let row = document.createElement('div');
    row.className = 'row';

    // Bild
    let imgDiv = document.createElement('div');
    imgDiv.className = 'image-container';

    let img = document.createElement('img');
    img.src = imageSrc;
    imgDiv.appendChild(img);

    // Chart Div (wichtig: eindeutige ID!)
    let chartDiv = document.createElement('div');
    chartDiv.className = 'chart-container';

    if (divName == "results"){
        if (imageSrc.includes("good")){
            row.className = 'row green';
        }
        else if (imageSrc.includes("bad")){
            row.className = 'row red';
        }
    }

    let chartId = 'chart-' + Date.now();
    chartDiv.id = chartId;

    // Row zusammenbauen
    row.appendChild(imgDiv);
    row.appendChild(chartDiv);
    container.appendChild(row);

    // Daten vorbereiten
    let labels = results.map(r => r.label);
    let values = results.map(r => r.confidence);

    let data = [{
        values: values,
        labels: labels,
        type: 'pie',
        //textinfo: 'label+percent'
    }];

    let layout = {
        height: 400,
        width: 600
    };

    // Plot zeichnen
    Plotly.newPlot(chartId, data, layout);
}

function loadDefaultImages() {
    defaultImages.forEach(src => {
        let img = new Image();
        img.src = src;

        img.onload = () => {
            classifier.classify(img, (results) => {
                createResultRow(src, results, 'results');
            });
        };
    });
}

