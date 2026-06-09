let customModel = null;
const charts = {};

// Ground Truth Funktion (aus Aufgabe angegegben)
function y(x) {
    return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
}
// Blur mit Box-Muller-Transformation https://mika-s.github.io/javascript/random/normal-distributed/2019/05/15/generating-normally-distributed-random-numbers-in-javascript.html
function boxMullerTransform() {
    let u = 0, v = 0;

    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();

    const z0 = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return z0;
}

function noise(variance) {
    const noisyboy = Math.sqrt(variance);
    return boxMullerTransform() * noisyboy;
}

function destroyChart(id) {
    if (charts[id]) {
        charts[id].destroy();
        delete charts[id];
    }
}

// datensatz speichern und laden

const DATA_KEY = 'ffnn_dataset_v1';

function saveDataset(d) {
    try { localStorage.setItem(DATA_KEY, JSON.stringify(d)); } catch (e) { }
}

function loadDataset() {
    try {
        const raw = localStorage.getItem(DATA_KEY);
        return raw ? JSON.parse(raw) : null;
    } catch (e) { return null; }
}

function genData() {
    // 100 random werte im bereich [-2, +2]
    const xs = Array.from({ length: 100 }, () => Math.random() * 4 - 2);
    // mappen/mischen jetzt nur die idicies um das ganze für die 50/50 Datenaufteilung so random wie möglich zu machen. (-0.5 damit zufällig positiv / negativ ist)
    const idx = xs.map((_, i) => i).sort(() => Math.random() - 0.5);

    // Daten in 50/50 aufteilen (Test und Trainingsdaten)
    const train = idx.slice(0, 50);
    const test = idx.slice(50);

    // returned den ganzen mist
    return {
        trainClean: make(xs, train, false),
        testClean: make(xs, test, false),
        trainNoisy: make(xs, train, true),
        testNoisy: make(xs, test, true),
    };
}

// Daten validieren

function validateDataset(d) {
    const log = [];
    let ok = true;

    function checkSplit(name, arr) {
        let nanCount = 0, oobCount = 0;
        for (const pt of arr) {
            if (!isFinite(pt.x) || !isFinite(pt.y)) nanCount++;
            if (pt.x < -2.001 || pt.x > 2.001) oobCount++;
        }
        const status = nanCount === 0 && oobCount === 0;
        if (!status) ok = false;
        log.push(`${status ? '✓' : '✗'} ${name}: N=${arr.length}, NaN/Inf=${nanCount}, x∉[-2,2]=${oobCount}`);
    }

    checkSplit('trainClean', d.trainClean);
    checkSplit('testClean', d.testClean);
    checkSplit('trainNoisy', d.trainNoisy);
    checkSplit('testNoisy', d.testNoisy);

    // Kein Überschneiden von Train/Test (via x-Werte-Vergleich)
    const txClean = new Set(d.trainClean.map(p => p.x.toFixed(10)));
    const overlap = d.testClean.filter(p => txClean.has(p.x.toFixed(10))).length;
    log.push(`${overlap === 0 ? '✓' : '✗'} Train/Test-Überschneidung: ${overlap} Punkte`);
    if (overlap > 0) ok = false;

    log.push(`→ Validierung ${ok ? 'bestanden' : 'fail'}`);
    return { ok, log: log.join('\n') };
}


function make(xs, ids, noisy) {
    const result = [];
    // geht jeden indicie durch (also 50 mal)
    for (let i = 0; i < ids.length; i++) {
        const index = ids[i];
        const xValue = xs[index];
        // hier wird y-Wert berechnet mit Ground Truth Formel
        const yClean = y(xValue);

        let yFinal;
        // mit rauschen oder ohne rauschen der Daten
        if (noisy) {
            yFinal = yClean + noise(0.05);
        } else {
            yFinal = yClean;
        }

        result.push({ x: xValue, y: yFinal });
    }

    return result;
}

// https://js.tensorflow.org/api/latest/#sequential
function createModel() {
    // Input Layer sind hintereinader
    const model = tf.sequential();

    // 2 hidden layers
    // 100 units wie in Aufgabe, relu activation tensor flow https://www.tensorflow.org/api_docs/python/tf/keras/activations/relu, 100 neuronen wie in aufgabenstellung
    model.add(tf.layers.dense({ inputShape: [1], units: 100, activation: 'relu' }));


    // kein Input Shape weil klar ist, dass die Inputs vom vorherigem layer sind.
    model.add(tf.layers.dense({ units: 100, activation: 'relu' }));

    // Output layer mit einem neuron
    model.add(tf.layers.dense({ units: 1 }));
    // Learning rate 0.01 und optimizer ADAM, siehe Aufgabe https://js.tensorflow.org/api/latest/#tf.LayersModel.compile
    model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });

    /* 
    batchSize: 32,
    */

    return model;
}

// Loss-History speichern und laden (für Verlustkurven nach Reload)
const LOSS_KEY = 'ffnn_losshistory_v1';

function saveLossHistories(h) {
    try { localStorage.setItem(LOSS_KEY, JSON.stringify(h)); } catch (e) { }
}

function loadLossHistories() {
    try {
        const r = localStorage.getItem(LOSS_KEY);
        return r ? JSON.parse(r) : {};
    } catch (e) { return {}; }
}

//zeichnet Loss pro Epoche auf
async function trainWithHistory(model, data, epochs, onProgress) {

    // Werte rausziehen
    const xValues = [];
    const yValues = [];
    for (let i = 0; i < data.length; i++) {
        xValues.push(data[i].x);
        yValues.push(data[i].y);
    }

    // In 2D Tensoren umwandeln
    const xs = tf.tensor2d(xValues, [xValues.length, 1]);
    const ys = tf.tensor2d(yValues, [yValues.length, 1]);

    const lossHistory = [];

    // Modell trainieren und sammelt Loss nach jeder Epoche
    await model.fit(xs, ys, {
        batchSize: 32,
        epochs: epochs,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                lossHistory.push(logs.loss);
                // Status alle 50 Epochen aktualisieren
                if (onProgress && (epoch % 50 === 0 || epoch === epochs - 1)) {
                    onProgress(epoch + 1, epochs, logs.loss);
                }
            }
        }
    });

    //speicher freihauen
    xs.dispose();
    ys.dispose();

    return lossHistory;
}

function mse(model, data) {

    // x und y Werte aus datensatz rausziehen
    const xValues = [];
    const yValues = [];

    for (let i = 0; i < data.length; i++) {
        xValues.push(data[i].x);
        yValues.push(data[i].y);
    }

    // räumt alle Tensoren die hier drin sind auf
    return tf.tidy(() => {

        // In 2D Tensoren umwandeln
        const xs = tf.tensor2d(xValues, [xValues.length, 1]);
        const ys = tf.tensor2d(yValues, [yValues.length, 1]);

        // Modell macht Vorhersagen für alle x-Werte
        const predictions = model.predict(xs);

        // Vergleicht Vorhersagen mit echten y-Werten und berechnet den mse
        const result = tf.losses.meanSquaredError(ys, predictions);

        // wandelt den Tensor zurück in eine normale Zahl
        return result.arraySync();
    });
}

function predLine(model) {
    // erzeugt x-Werte die gleichmäßig verteilt sind in Bereich [-2, +2]
    // abgeletet aus Tutorial
    return tf.tidy(() => {

        const xs = tf.linspace(-2, 2, 200);
        const ys = model.predict(xs.reshape([200, 1]));

        return { xs: Array.from(xs.dataSync()), ys: Array.from(ys.dataSync()) };
    });
}

// Chart.js graphen

// so sehen die graphen aus
const opts = {
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: { legend: { display: false } },
    scales: {
        x: { type: 'linear', min: -2.1, max: 2.1, grid: { color: '#f0f0f0' }, ticks: { color: '#aaa', font: { size: 10 }, maxTicksLimit: 5 } },
        y: { grid: { color: '#f0f0f0' }, ticks: { color: '#aaa', font: { size: 10 }, maxTicksLimit: 5 } }
    }
};

// sucht id und scatter plot zeichnen in
// blau: Trainingsdaten 
// rot: Testdaten
function scatter(id, a, b) {
    let chartStatus = Chart.getChart(document.getElementById(id)); // <canvas> id
    if (chartStatus != undefined) {
        chartStatus.destroy();
    }
    try {
        new Chart(document.getElementById(id), {
            type: 'scatter',
            data: {
                datasets: [
                    { data: a.map(d => ({ x: d.x, y: d.y })), backgroundColor: '#16fa2188', pointRadius: 3 },
                    { data: b.map(d => ({ x: d.x, y: d.y })), backgroundColor: '#ff44cd88', pointRadius: 3 }
                ]
            },
            options: opts
        });
    } catch (e) {
        new Chart(document.getElementById(id), {
            type: 'scatter',
            data: {
                datasets: [
                    { data: a.map(d => ({ x: d.x, y: d.y })), backgroundColor: '#16fa2188', pointRadius: 3 },
                    { data: b.map(d => ({ x: d.x, y: d.y })), backgroundColor: '#ff44cd88', pointRadius: 3 }
                ]
            },
            options: opts
        });
    }

}

// nur linie keine scatter daten
function predChart(id, pts, line) {
    try {
        new Chart(document.getElementById(id), {
            type: 'scatter',
            data: {
                datasets: [
                    { data: pts.map(d => ({ x: d.x, y: d.y })), backgroundColor: '#0ba0e188', pointRadius: 3 },
                    { data: line.xs.map((x, i) => ({ x, y: line.ys[i] })), type: 'line', borderColor: '#111', borderWidth: 1.5, pointRadius: 0 }
                ]
            },
            options: opts
        });
    }
    catch (e) {
        let chartStatus = Chart.getChart(document.getElementById(id)); // <canvas> id
        if (chartStatus != undefined) {
            chartStatus.destroy();
        }
        new Chart(document.getElementById(id), {
            type: 'scatter',
            data: {
                datasets: [
                    { data: pts.map(d => ({ x: d.x, y: d.y })), backgroundColor: '#0ba0e188', pointRadius: 3 },
                    { data: line.xs.map((x, i) => ({ x, y: line.ys[i] })), type: 'line', borderColor: '#111', borderWidth: 1.5, pointRadius: 0 }
                ]
            },
            options: opts
        });
    }

}

// Loss-Kurve über Epochen
function lossChart(id, history) {
    // Auf max 200 Punkte reduziert
    const step = Math.max(1, Math.floor(history.length / 200));

    const labels = [];
    const data = [];

    for (let i = 0; i < history.length; i += step) {

        labels.push(i);
        data.push(history[i]);
    }

    let chartStatus = Chart.getChart(document.getElementById(id)); // <canvas> id
    if (chartStatus != undefined) {
        chartStatus.destroy();
    }

    new Chart(document.getElementById(id), {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                borderColor: '#0ba0e188',
                borderWidth: 1.5,
                pointRadius: 0,
                fill: false,
                tension: 0
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    grid: { color: '#f0f0f0' },
                    ticks: { color: '#aaa', font: { size: 10 }, maxTicksLimit: 6 },
                    title: { display: true, text: 'Epoche', color: '#aaa', font: { size: 10 } }
                },
                y: {
                    grid: { color: '#f0f0f0' },
                    ticks: { color: '#aaa', font: { size: 10 }, maxTicksLimit: 5 },
                    title: { display: true, text: 'MSE', color: '#aaa', font: { size: 10 } }
                }
            }
        }
    });
}

//Buttons de/aktivieren während Training läuft
function setBtns(disabled) {
    ['btn-run', 'btn-load', 'btn-clear', 'btn-custom', 'btn-save-custom', 'btn-load-custom'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.disabled = disabled;
    });
}

// Validierungsergebnis
function showValidation(result) {

    const el = document.getElementById('validation-log');
    if (!el) return;

    el.textContent = result.log;

    el.style.borderColor = result.ok ? '#bbf7d0' : '#fca5a5';
    el.style.background = result.ok ? '#f0fdf4' : '#fef2f2';
}


async function getOrTrain(key, trainData, epochs, label, statusEl) {
    let m = null;

    try { m = await tf.loadLayersModel('indexeddb://' + key); } catch (e) { }

    if (!m) {
        m = createModel();

        const history = await trainWithHistory(m, trainData, epochs, (epoch, total, loss) => {
            if (statusEl) statusEl.textContent = `Training: ${label} (${epoch}/${total} Epochen, Loss: ${loss.toFixed(5)})...`;
        });
        try { await m.save('indexeddb://' + key); } catch (e) { }
        return { model: m, history, fresh: true };
    }
    return { model: m, history: null, fresh: false };
}

// true werden neue Daten gemacht
// Neutraining false werden gespeicherte laden
async function run(forceNew = false) {
    const status = document.getElementById('status');
    setBtns(true);

    if (forceNew) {
        for (const key of ['clean', 'best', 'over']) {
            try { await tf.io.removeModel('indexeddb://' + key); } catch (e) { }
        }
        localStorage.removeItem(DATA_KEY);
        localStorage.removeItem(LOSS_KEY);
    }

    //Datensatz laden und speichern
    status.textContent = 'Daten vorbereiten...';
    let d = loadDataset();
    if (!d) { d = genData(); saveDataset(d); }

    //Daten validieren bevor irgendwas gezeichnet wird
    const valResult = validateDataset(d);

    showValidation(valResult);

    if (!valResult.ok) {
        status.textContent = 'Datenvalidierung fehlgeschlagen!';
        setBtns(false);
        return;
    }

    // hier einfach nur Test und Trainingsdaten gerendert
    scatter('c1', d.trainClean, d.testClean);
    scatter('c2', d.trainNoisy, d.testNoisy);

    // Loss-Histories laden
    let savedHistories = loadLossHistories();

    // Clean Model
    status.textContent = 'Clean-Modell vorbereiten...';

    const { model: mClean, history: hClean, fresh: fClean } = await getOrTrain('clean', d.trainClean, 500, 'Clean', status);

    //History speichern
    if (fClean) { savedHistories.clean = hClean; saveLossHistories(savedHistories); }

    const lineC = predLine(mClean);

    predChart('c3', d.trainClean, lineC);
    predChart('c4', d.testClean, lineC);

    document.getElementById('m1').textContent = mse(mClean, d.trainClean).toFixed(6);
    document.getElementById('m2').textContent = mse(mClean, d.testClean).toFixed(6);

    if (savedHistories.clean) lossChart('lc1', savedHistories.clean);

    // Best Case Model
    status.textContent = 'Best-Fit-Modell vorbereiten...';

    const { model: mBest, history: hBest, fresh: fBest } = await getOrTrain('best', d.trainNoisy, 300, 'Best-Fit', status);

    if (fBest) { savedHistories.best = hBest; saveLossHistories(savedHistories); }
    const lineB = predLine(mBest);


    predChart('c5', d.trainNoisy, lineB);
    predChart('c6', d.testNoisy, lineB);

    document.getElementById('m3').textContent = mse(mBest, d.trainNoisy).toFixed(6);
    document.getElementById('m4').textContent = mse(mBest, d.testNoisy).toFixed(6);

    if (savedHistories.best) lossChart('lc2', savedHistories.best);

    // Overfitted Model
    status.textContent = 'Overfit-Modell vorbereiten...';

    const { model: mOver, history: hOver, fresh: fOver } = await getOrTrain('over', d.trainNoisy, 4000, 'Overfit', status);

    if (fOver) { savedHistories.over = hOver; saveLossHistories(savedHistories); }
    const lineO = predLine(mOver);



    predChart('c7', d.trainNoisy, lineO);
    predChart('c8', d.testNoisy, lineO);



    document.getElementById('m5').textContent = mse(mOver, d.trainNoisy).toFixed(6);
    document.getElementById('m6').textContent = mse(mOver, d.testNoisy).toFixed(6);


    if (savedHistories.over) lossChart('lc3', savedHistories.over);

    status.textContent = 'Fertig.';
    setBtns(false);
}

// Alles löschen
/*
async function clearAll() {

    if (!confirm('Alle gespeicherten Daten und Modelle löschen?')) return;
    setBtns(true);

    localStorage.removeItem(DATA_KEY);
    localStorage.removeItem(LOSS_KEY);

    for (const key of ['clean', 'best', 'over']) {
        try { await tf.io.removeModel('indexeddb://' + key); } catch (e) { }
    }


    const status = document.getElementById('status');


    if (status) status.textContent = 'Alles gelöscht. Klicke "Neu starten" zum Trainieren.';

    const valEl = document.getElementById('validation-log');
    if (valEl) valEl.textContent = '—';
    setBtns(false);
}
*/
//Buttons beim Laden gespeicherte Daten verwenden
document.addEventListener('DOMContentLoaded', () => {
    const btnRun = document.getElementById('btn-run');
    const btnLoad = document.getElementById('btn-load');
    const btnClear = document.getElementById('btn-clear');
    const btnCustom = document.getElementById('btn-custom');
    const btnSaveCustom = document.getElementById('btn-save-custom');
    const btnLoadCustom = document.getElementById('btn-load-custom');

    if (btnRun) btnRun.addEventListener('click', () => run(true));
    if (btnLoad) btnLoad.addEventListener('click', () => run(false));
    if (btnClear) btnClear.addEventListener('click', clearAll);
    if (btnCustom) btnCustom.addEventListener('click', trainCustomModel);
    if (btnSaveCustom) btnSaveCustom.addEventListener('click', saveCustomModel);
    if (btnLoadCustom) btnLoadCustom.addEventListener('click', loadCustomModel);

    // Beim ersten Laden nur gespeicherte Daten verwenden werden nicht neu generieren
    run(false);
});

async function trainCustomModel() {
    const status = document.getElementById('status');

    //Parameter aus den Eingabefeldern lesen
    const epochs = parseInt(document.getElementById('custom-epochs').value);
    const lr = parseFloat(document.getElementById('custom-lr').value);
    const useNoisy = document.getElementById('custom-data').value === 'noisy';

    //Prüfen ob ein Datensatz vorhanden ist
    const d = loadDataset();
    if (!d) {
        status.textContent = 'Kein Datensatz vorhanden — erst "Neu starten" klicken!';
        return;
    }

    //Richtigen Datensatz wählen 
    const trainData = useNoisy ? d.trainNoisy : d.trainClean;
    const testData = useNoisy ? d.testNoisy : d.testClean;

    //Buttons sperren damit nicht doppelt geklickt wird
    setBtns(true);
    status.textContent = `Eigenes Modell wird trainiert (${epochs} Epochen, lr=${lr})...`;

    //Neues Modell mit gewählter Lernrate createn
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [1], units: 100, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1 }));
    model.compile({ optimizer: tf.train.adam(lr), loss: 'meanSquaredError' });

    //trainWithHistory für die Loss-Kurve
    const history = await trainWithHistory(model, trainData, epochs, (epoch, total, loss) => {
        status.textContent = `Eigenes Modell: ${epoch}/${total} Epochen, Loss: ${loss.toFixed(5)}`;
    });

    // Modell global speichern damit btn-save-custom drauf zugreifen kann
    customModel = model;

    //Ergebnis-Bereich einblenden
    document.getElementById('custom-result').style.display = 'block';

    // Vorhersage-Linie berechnen und Diagramme generieren
    const line = predLine(model);
    predChart('cc1', trainData, line);
    predChart('cc2', testData, line);

    // MSE berechnen und anzeigen
    document.getElementById('cm1').textContent = mse(model, trainData).toFixed(6);
    document.getElementById('cm2').textContent = mse(model, testData).toFixed(6);

    //Loss-Kurve zeichnen
    lossChart('lcc1', history);

    status.textContent = `Eigenes Modell fertig. Train MSE: ${mse(model, trainData).toFixed(6)} | Test MSE: ${mse(model, testData).toFixed(6)}`;
    setBtns(false);
}


//Eigenes Modell speicher
async function saveCustomModel() {
    const status = document.getElementById('status');

    if (!customModel) {
        status.textContent = 'Kein eigenes Modell vorhanden, bitte erst trainieren!';
        return;
    }

    try {
        await customModel.save('indexeddb://custom-model');
        status.textContent = 'Eigenes Modell gespeichert.';
    } catch (err) {
        console.error(err);
        status.textContent = 'Fehler beim Speichern.';
    }
}


//Eigenes Modell laden
async function loadCustomModel() {
    const status = document.getElementById('status');
    const d = loadDataset();

    try {
        // Aus IndexedDB laden
        customModel = await tf.loadLayersModel('indexeddb://custom-model');

        // Ergebnis-Bereich einblenden
        document.getElementById('custom-result').style.display = 'block';

        // Vorhersage neu zeichnen wenn Datensatz vorhanden
        if (d) {
            const line = predLine(customModel);
            predChart('cc1', d.trainNoisy, line);
            predChart('cc2', d.testNoisy, line);
            document.getElementById('cm1').textContent = mse(customModel, d.trainNoisy).toFixed(6);
            document.getElementById('cm2').textContent = mse(customModel, d.testNoisy).toFixed(6);
        }

        status.textContent = 'Eigenes Modell geladen.';
    } catch (e) {
        status.textContent = 'Kein eigenes Modell gefunden.';
    }
}