const CONFIG = {
  SEQ_LEN: 5,          // Länge der Eingabesequenz (Kontextfenster)
  EMBED_DIM: 64,       // Dimension der Embedding-Schicht
  LSTM_UNITS: 100,     // Units pro LSTM-Layer (Vorgabe: 100)
  LEARNING_RATE: 0.01, // Vorgabe: 0.01 oder 0.001
  BATCH_SIZE: 32,      // Vorgabe: 32
  EPOCHS: 1           // zum Ausprobieren, Loss beobachten
};

let vocab = {}, vocabInv = {}, vocabSize = 0;
let xs = null, ys = null;
let model = null;
let stopTrain = false, stopAuto = false;
let lossHistory = [], accHistory = [];

// Helpers
const $ = id => document.getElementById(id);
function log(msg) {
  $('log').innerHTML += msg + '<br>';
  $('log').scrollTop = $('log').scrollHeight;
}
function tokenize(text) {
  return text.toLowerCase().replace(/[^a-zäöüß\s]/g, ' ').trim().split(/\s+/).filter(Boolean);
}

// Auto-Load pretrained model from GitHub Pages
(async () => {
  try {
    const vocabResp = await fetch('dataset-test.txt');
    if (!vocabResp.ok) throw new Error('Kein Vokabular gefunden');
    const vocabData = await vocabResp.json();
    vocab = vocabData.vocab;
    vocabInv = vocabData.vocabInv;
    vocabSize = vocabData.vocabSize;

    model = await tf.loadLayersModel('dataset-test.txt');
    log('✓ Vortrainiertes Modell geladen.');
    $('statusBanner').textContent = '✓ Vortrainiertes Modell geladen — bereit zur Vorhersage.';
    enablePredictButtons();
    $('btnExport').disabled = false;
    $('btnBuild').disabled = false;
    $('btnEval').disabled = false;
  } catch (e) {
    $('statusBanner').textContent = 'Kein vortrainiertes Modell gefunden — bitte tokenisieren, bauen und trainieren.';
    $('btnBuild').disabled = true; // enabled after tokenize
  }
})();

function enablePredictButtons() {
  $('btnPredict').disabled = false;
  $('btnNext').disabled = false;
  $('btnAuto').disabled = false;
  $('btnReset').disabled = false;
}
/*
// Tokenisierung
$('btnProcess').onclick = () => {
  const words = tokenize($('corpus').value);
  if (words.length < 10) { log('Zu wenig Text.'); return; }

  vocab = {}; let idx = 1;
  for (const w of words) if (!vocab[w]) vocab[w] = idx++;
  vocabInv = Object.fromEntries(Object.entries(vocab).map(([w, i]) => [i, w]));
  vocabSize = idx;

  const seqLen = CONFIG.SEQ_LEN;
  const tokens = words.map(w => vocab[w]);
  const seqs = [];
  for (let i = seqLen; i < tokens.length; i++)
    seqs.push(tokens.slice(i - seqLen, i + 1));

  if (xs) xs.dispose(); if (ys) ys.dispose();
  xs = tf.tensor2d(seqs.map(s => s.slice(0, seqLen)), [seqs.length, seqLen], 'int32');
  ys = tf.oneHot(tf.tensor1d(seqs.map(s => s[seqLen]), 'int32'), vocabSize).toFloat();

  log(`Vokabular: ${vocabSize - 1} Wörter | Sequenzen: ${seqs.length}`);
  $('btnBuild').disabled = false;
};
*/
// Modell (Stacked LSTM)
/*
$('btnBuild').onclick = () => {
  if (model) model.dispose();
  const seqLen = CONFIG.SEQ_LEN;
  const embedDim = CONFIG.EMBED_DIM;
  const lstmUnits = CONFIG.LSTM_UNITS;
  const lr = CONFIG.LEARNING_RATE;

  model = tf.sequential({
    layers: [
      tf.layers.embedding({ inputDim: vocabSize, outputDim: embedDim, inputLength: seqLen }),
      tf.layers.lstm({ units: lstmUnits, returnSequences: true }),
      tf.layers.lstm({ units: lstmUnits }),
      tf.layers.dense({ units: vocabSize, activation: 'softmax' })
    ]
  });

  model.compile({
    optimizer: tf.train.adam(lr),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  log(`Modell: Embedding(${embedDim}) → LSTM(${lstmUnits}) → LSTM(${lstmUnits}) → Dense(${vocabSize})`);
  log(`Optimizer: Adam (lr=${lr}) | Batch-Size: ${CONFIG.BATCH_SIZE} | Epochs: ${CONFIG.EPOCHS}`);
  $('btnTrain').disabled = false;
};
*/
// Training
$(document).ready(function () {
  $('btnTrain').onclick = async () => {
    stopTrain = false;
    lossHistory = []; accHistory = [];
    $('btnStopTrain').disabled = false;
    $('btnTrain').disabled = true;
    console.log("This is training!");

    const epochs = CONFIG.EPOCHS;
    const batchSize = CONFIG.BATCH_SIZE;

    console.log("Start Training!");

    await model.fit(xs, ys, {
      epochs, batchSize, shuffle: true,
      callbacks: {
        onEpochEnd: (ep, logs) => {
          const acc = logs.acc ?? logs.accuracy ?? 0;
          lossHistory.push(logs.loss);
          accHistory.push(acc);
          updateChart();
          if ((ep + 1) % 5 === 0 || ep === 0)
            log(`Epoch ${ep + 1}/${epochs} — loss: ${logs.loss.toFixed(4)} | acc: ${acc.toFixed(3)}`);
          if (stopTrain) model.stopTraining = true;
        },
        onTrainEnd: () => {
          log('Training abgeschlossen.');
          $('btnTrain').disabled = false;
          $('btnStopTrain').disabled = true;
          $('btnExport').disabled = false;
          $('btnEval').disabled = false;
          enablePredictButtons();
          predict();
        }
      }
    });
  };
  $('btnStopTrain').onclick = () => { stopTrain = true; };
});
// Loss Chart
function updateChart() {
  const epochs = lossHistory.map((_, i) => i + 1);
  Plotly.react('lossChart', [
    { x: epochs, y: lossHistory, name: 'Loss', type: 'scatter', mode: 'lines', line: { color: '#333' } },
    { x: epochs, y: accHistory, name: 'Accuracy', type: 'scatter', mode: 'lines', line: { color: '#888', dash: 'dot' }, yaxis: 'y2' }
  ], {
    margin: { t: 10, r: 50, b: 30, l: 50 },
    xaxis: { title: 'Epoch', tickfont: { size: 10 } },
    yaxis: { title: 'Loss', tickfont: { size: 10 } },
    yaxis2: { title: 'Accuracy', overlaying: 'y', side: 'right', tickfont: { size: 10 }, range: [0, 1] },
    legend: { font: { size: 10 }, x: 0.5, y: 1.1, orientation: 'h' },
    font: { family: 'monospace', size: 11 }
  }, { displayModeBar: false });
}

// Export Modell + Vokabular
$('btnExport').onclick = async () => {
  if (!model) { log('Kein Modell vorhanden.'); return; }

  // Modell als downloads://lstm-lm speichern
  await model.save('downloads://lstm-lm');

  // Vokabular als JSON herunterladen
  const vocabData = JSON.stringify({ vocab, vocabInv, vocabSize });
  const blob = new Blob([vocabData], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'vocab.json';
  a.click();

  log('✓ Exportiert: lstm-lm.json, lstm-lm.weights.bin, vocab.json');
  log('→ Dateien in repo/ ablegen: model/model.json, model/group1-shard1of1.bin, vocab.json');
};

// Top-k Evaluation & Perplexity
$('btnEval').onclick = async () => {
  if (!model || !xs) { log('Modell oder Daten fehlen.'); return; }
  log('Berechne Top-k Accuracy…');

  const seqLen = CONFIG.SEQ_LEN;
  const xArr = xs.arraySync();
  const yArr = ys.arraySync(); // one-hot

  const trueLabels = yArr.map(row => row.indexOf(Math.max(...row)));
  const inp = tf.tensor2d(xArr, [xArr.length, seqLen], 'int32');
  const probsAll = model.predict(inp).arraySync();
  inp.dispose();

  const ks = [1, 5, 10, 20];
  const counts = { 1: 0, 5: 0, 10: 0, 20: 0 };
  let totalLogProb = 0;

  for (let i = 0; i < probsAll.length; i++) {
    const probs = probsAll[i];
    const trueIdx = trueLabels[i];
    totalLogProb += Math.log(Math.max(probs[trueIdx], 1e-10));
    const sorted = probs.map((p, j) => [j, p]).sort((a, b) => b[1] - a[1]);
    for (const k of ks) {
      const topK = sorted.slice(0, k).map(x => x[0]);
      if (topK.includes(trueIdx)) counts[k]++;
    }
  }

  const n = probsAll.length;
  const perplexity = Math.exp(-totalLogProb / n);

  for (const k of ks) {
    const acc = (counts[k] / n * 100).toFixed(1) + '%';
    const el = $('acc' + k);
    if (el) el.textContent = acc;
  }
  $('perplexityText').textContent = `Perplexity: ${perplexity.toFixed(2)} (niedriger = besser; Zufallsmodell hätte PP ≈ ${vocabSize})`;
  log(`Top-k Accuracy berechnet. Perplexity: ${perplexity.toFixed(2)}`);
};

// Vorhersage
function predict() {
  if (!model) return;
  const seqLen = CONFIG.SEQ_LEN;
  const raw = tokenize($('promptText').value || 'the');
  const context = raw.slice(-seqLen).map(w => vocab[w] ?? 0);
  while (context.length < seqLen) context.unshift(0);

  const inp = tf.tensor2d([context], [1, seqLen], 'int32');
  const probs = model.predict(inp).arraySync()[0];
  inp.dispose();

  const top = probs.map((p, i) => [i, p]).filter(([i]) => i > 0)
    .sort((a, b) => b[1] - a[1]).slice(0, 5);

  const box = $('predictions');
  box.innerHTML = '';
  top.forEach(([i, p]) => {
    const word = vocabInv[i] ?? '?';
    const chip = document.createElement('div');
    chip.className = 'chip';
    chip.innerHTML = `${word}<span class="prob">${(p * 100).toFixed(1)}%</span>`;
    chip.onclick = () => appendWord(word);
    box.appendChild(chip);
  });
  return top;
}

function appendWord(word) {
  const el = $('promptText');
  el.value = (el.value.trim() + ' ' + word).trim();
  predict();
}

$('btnPredict').onclick = predict;

$('btnNext').onclick = () => {
  const top = predict();
  if (top && top.length > 0) appendWord(vocabInv[top[0][0]]);
};

$('btnAuto').onclick = async () => {
  stopAuto = false;
  $('btnStopAuto').disabled = false;
  $('btnAuto').disabled = true;
  for (let i = 0; i < 10 && !stopAuto; i++) {
    const top = predict();
    if (!top || top.length === 0) break;
    appendWord(vocabInv[top[0][0]]);
    await new Promise(r => setTimeout(r, 400));
  }
  $('btnAuto').disabled = false;
  $('btnStopAuto').disabled = true;
};
$('btnStopAuto').onclick = () => { stopAuto = true; };

$('btnReset').onclick = () => {
  $('promptText').value = '';
  $('predictions').innerHTML = '';
  if (model) { model.dispose(); model = null; }
  $('btnPredict').disabled = true;
  $('btnNext').disabled = true;
  $('btnAuto').disabled = true;
  $('btnReset').disabled = true;
  $('btnBuild').disabled = false;
  $('btnTrain').disabled = true;
  $('btnExport').disabled = true;
  $('btnEval').disabled = true;
  $('statusBanner').textContent = 'Modell zurückgesetzt.';
  log('Zurückgesetzt.');
};
