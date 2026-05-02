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
    //classifier = ml5.imageClassifier("MobileNet");
    img = loadImage("images/good1.jpg");

    //loadDefaultImages(); // HIER aufrufen!
}

function setup() {
    //createCanvas(400, 400);

    //classifier.classify(img, gotResult);
    //image(img, 300, 300);
    // Drag and Drop
    let dropArea = createCanvas(710, 400);
    dropArea.parent('canvas-wrapper');
    dropArea.drop(gotFile);
    noLoop();


    classifier = ml5.imageClassifier("MobileNet", () => {
        loadDefaultImages();
    });
    // Default-Bilder laden
    //loadDefaultImages();
}

function draw() {
    background(245, 244, 240);
    fill(100);
    noStroke();
    textSize(13);
    textAlign(CENTER);
    textFont('IBM Plex Mono');
    text(canvasText, width / 2, height / 2);
    describe(`Canvas mit dem Text "${canvasText}" in der Mitte.`);
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

    Plotly.newPlot('myDiv', data, layout);
}

function gotFile(file) {
    if (file.type === 'image') {
        let imgElement = createImg(file.data, '').hide();
        let done = false;
        classifier.classifyStart(imgElement.elt, (results) => {
            if (done) return;
            done = true;
            classifier.classifyStop();
            createResultRow(file.data, results, 'uploadedImages');
        });
    } else {
        canvasText = 'Not an image file!';
        redraw();
    }
}

function createResultRow(imageSrc, results, divName) {
    let container = document.getElementById(divName);

    let row = document.createElement('div');
    row.className = 'row';

    let imgDiv = document.createElement('div');
    imgDiv.className = 'image-container';

    let imgEl = document.createElement('img');
    imgEl.src = imageSrc;
    imgDiv.appendChild(imgEl);

    let chartDiv = document.createElement('div');
    chartDiv.className = 'chart-container';

    if (divName === "results") {
        row.className = imageSrc.includes("good") ? 'row green' : 'row red';
    }

    let chartId = 'chart-' + Date.now();
    chartDiv.id = chartId;

    row.appendChild(imgDiv);
    row.appendChild(chartDiv);
    container.appendChild(row);

    let labels = results.map(r => r.label);
    let values = results.map(r => r.confidence);
    let data = [{ values, labels, type: 'pie' }];

    let colorway;
    if (imageSrc.includes("good")) {
        colorway = ['#2d6a4f', '#52b788', '#95d5b2', '#b7e4c7', '#d8f3dc', '#f0faf3'];
    } else if (imageSrc.includes("bad")) {
        colorway = ['#7d0000', '#c1121f', '#e63946', '#f4a261', '#f8c8a0', '#fde8d8'];
    } else {
        colorway = ['#1a1a1a', '#555', '#888', '#aaa', '#ccc', '#e0e0e0'];
    }

    let localLayout = {
        height: 260,
        autosize: true,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#1a1a1a', family: 'IBM Plex Mono', size: 11 },
        margin: { t: 10, b: 10, l: 10, r: 10 },
        showlegend: true,
        legend: { font: { size: 10 }, orientation: 'v' },
        colorway: colorway
    };

    Plotly.newPlot(chartId, data, localLayout, { responsive: true, displayModeBar: false });
}

function loadDefaultImages() {
    let i = 0;

    function classifyNext() {
        if (i >= defaultImages.length) return;

        let src = defaultImages[i];
        i++;

        let imgEl = new Image();
        imgEl.src = src;
        imgEl.onload = () => {
            let done = false;
            classifier.classifyStart(imgEl, (results) => {
                if (done) return;
                done = true;
                classifier.classifyStop();
                const target = src.includes("good") ? 'results-good' : 'results-bad';
                createResultRow(src, results, target);
                classifyNext(); // erst wenn fertig, nächstes Bild
            });
        };
    }

    classifyNext();
}

