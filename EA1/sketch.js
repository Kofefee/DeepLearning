let classifier;

let images = [
    { src: "images/good1.jpg", label: "", conf: 0 },
    { src: "images/good2.jpg", label: "", conf: 0 },
    { src: "images/good3.jpg", label: "", conf: 0 },
    { src: "images/bad1.jpg", label: "", conf: 0 },
    { src: "images/bad2.jpg", label: "", conf: 0 },
    { src: "images/bad3.jpg", label: "", conf: 0 }
];

let loadedImages = [];

let canvasText = 'Drag an image file onto the canvas.';

function preload() {
    classifier = ml5.imageClassifier("MobileNet");

    for (let i = 0; i < images.length; i++) {
        loadedImages[i] = loadImage(images[i].src);
    }
}

function setup() {
    createCanvas(400, 400);

    // Klassifiziere alle Bilder
    for (let i = 0; i < loadedImages.length; i++) {
        classifyImage(i);
    }

    // Drag and Drop
    let dropArea = createCanvas(710, 400);
    dropArea.drop(gotFile);
    //noLoop();
}

function classifyImage(index) {
    classifier.classify(loadedImages[index], (err, results) => {
        if (err) {
            console.error(err);
            return;
        }

        images[index].label = results[0].label;
        images[index].conf = results[0].confidence;

        drawPieChart(index);
    });
}

function draw() {
    background(240);

    for (let i = 0; i < loadedImages.length; i++) {

        let y = i * 100;

        // Bild links
        image(loadedImages[i], 10, y, 80, 80);

        // Text
        fill(0);
        textSize(12);
        text(images[i].label, 100, y + 30);
        text(nf(images[i].conf * 100, 2, 2) + "%", 100, y + 50);
    }
}

function drawPieChart(index) {

    let conf = images[index].conf;

    let data = [{
        values: [conf, 1 - conf],
        labels: ['Confidence', 'Rest'],
        type: 'pie'
    }];

    let layout = {
        height: 100,
        width: 150
    };

    let divId = "chart" + index;

    if (!document.getElementById(divId)) {
        let div = createDiv().id(divId);
        div.position(300, index * 100);
    }

    Plotly.newPlot(divId, data, layout);
}

// Upload Funktion
function gotFile(file) {

    if (file.type === 'image') {

        let img = createImg(file.data, '').hide();

        classifier.classify(img, (err, results) => {
            if (err) return;

            let label = results[0].label;
            let conf = results[0].confidence;

            // Anzeige
            let div = createDiv("Upload Ergebnis: " + label + " (" + nf(conf * 100, 2, 2) + "%)");
            div.position(10, 620);

            let data = [{
                values: [conf, 1 - conf],
                labels: ['Confidence', 'Rest'],
                type: 'pie'
            }];

            Plotly.newPlot(div.elt, data);
        });

    }
}