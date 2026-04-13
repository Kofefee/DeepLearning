let classifier;

let images = [
    { src: "images/good1.jpg", label: "", conf: 0, type: "good" },
    { src: "images/good2.jpg", label: "", conf: 0, type: "good" },
    { src: "images/good3.jpg", label: "", conf: 0, type: "good" },
    { src: "images/bad1.jpg", label: "", conf: 0, type: "bad" },
    { src: "images/bad2.jpg", label: "", conf: 0, type: "bad" },
    { src: "images/bad3.jpg", label: "", conf: 0, type: "bad" }
];

let loadedImages = [];
let canvas;

function preload() {
    classifier = ml5.imageClassifier("MobileNet");

    for (let i = 0; i < images.length; i++) {
        loadedImages[i] = loadImage(images[i].src);
    }
}

function setup() {
    canvas = createCanvas(900, 700);
    canvas.drop(gotFile);

    // Klassifizieren
    for (let i = 0; i < loadedImages.length; i++) {
        classifyImage(i);
    }
}

function classifyImage(index) {
    classifier.classify(loadedImages[index], (err, results) => {
        if (err) return;

        images[index].label = results[0].label;
        images[index].conf = results[0].confidence;

        drawPieChart(index);
    });
}

function draw() {
    background(240);

    textSize(16);
    fill(0);

    text("Gute Beispiele", 50, 30);
    text("Schlechte Beispiele", 50, 360);

    for (let i = 0; i < loadedImages.length; i++) {

        let isGood = images[i].type === "good";

        let row = isGood ? i : i - 3;
        let yOffset = isGood ? 50 : 380;

        let y = yOffset + row * 100;

        // Bild links
        image(loadedImages[i], 20, y, 80, 80);

        // Text daneben
        fill(0);
        textSize(12);
        text(images[i].label, 120, y + 30);
        text(nf(images[i].conf * 100, 2, 2) + "%", 120, y + 50);
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
        width: 150,
        margin: { t: 10, b: 10, l: 10, r: 10 }
    };

    let isGood = images[index].type === "good";
    let row = isGood ? index : index - 3;
    let yOffset = isGood ? 50 : 380;

    let y = yOffset + row * 100;

    let divId = "chart" + index;

    if (!document.getElementById(divId)) {
        let div = createDiv().id(divId);
        div.position(350, y);
    }

    Plotly.newPlot(divId, data, layout);
}

// 📤 Upload Funktion
function gotFile(file) {
    if (file.type === 'image') {

        let img = createImg(file.data, '').hide();

        classifier.classify(img, (err, results) => {
            if (err) return;

            let label = results[0].label;
            let conf = results[0].confidence;

            let y = 650;

            let resultText = createDiv(
                "Upload: " + label + " (" + nf(conf * 100, 2, 2) + "%)"
            );
            resultText.position(20, y);

            let data = [{
                values: [conf, 1 - conf],
                labels: ['Confidence', 'Rest'],
                type: 'pie'
            }];

            let div = createDiv();
            div.position(350, y);

            Plotly.newPlot(div.elt, data, {
                height: 120,
                width: 180
            });
        });
    }
}