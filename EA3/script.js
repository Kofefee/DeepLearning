async function hello() {
  var vocabResp = await fetch('german-dataset.txt');
  let text = await vocabResp.text();
  console.log("Start Training!");

  text = text

  .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .replace(/\ufeff/g, '')
    .replace(/\t/g, ' ')
    .replace(/ +/g, ' ')
    .replace(/\n{2,}/g, '\n')
    .replace(/"/g, '')
    .trim();

  text = text.split(/\s+/).filter(w => w.length > 0).join(' ');

  var words = text.split(' ');
  var wordCounts = {};
  words.forEach(w => { wordCounts[w] = (wordCounts[w] || 0) + 1; });

  var sortedWords = Object.keys(wordCounts).sort((a, b) => wordCounts[b] - wordCounts[a]);
  var wordIndexLocal = {};
  sortedWords.forEach((w, i) => { wordIndexLocal[w] = i + 1; });
  localStorage.setItem('tokenizer', JSON.stringify(wordIndexLocal));

  downloadTokenizer(wordIndexLocal);

  var sequenceData = words.map(w => wordIndexLocal[w]);
  var vocabularySize = Object.keys(wordIndexLocal).length + 1;

  var sequence = [];
  for (let i = 3; i < sequenceData.length; i++) {
    sequence.push(sequenceData.slice(i - 3, i + 1));
  }

  console.log("Vocab size:", vocabularySize);

  var xArr = sequence.map(s => s.slice(0, 3));
  var yArr = sequence.map(s => s[3]);

  console.log("Anzahl Samples:", xArr.length);

  var x = tf.tensor2d(xArr);
  var y = tf.tensor1d(yArr, 'float32');

  var SEQ_LEN = 3;
  var EMBED_DIM = 64;
  var LSTM_UNITS = 100;
  var LEARNING_RATE = 0.01;
  var BATCH_SIZE = 32;
  var EPOCHS = 2;

  var trainedModel = tf.sequential({
    layers: [
      tf.layers.embedding({
        inputDim: vocabularySize,
        outputDim: EMBED_DIM,
        inputLength: SEQ_LEN
      }),
      tf.layers.lstm({ units: LSTM_UNITS, returnSequences: true }),
      tf.layers.lstm({ units: LSTM_UNITS }),
      tf.layers.dense({ units: vocabularySize, activation: 'softmax' })
    ]
  });

  trainedModel.compile({
    optimizer: tf.train.adam(LEARNING_RATE),
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy']
  });

  trainedModel.summary();

  let bestLoss = Infinity;
  await trainedModel.fit(x, y, {
    epochs: EPOCHS,
    batchSize: BATCH_SIZE,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
        if (logs.loss < bestLoss) {
          bestLoss = logs.loss;
          await trainedModel.save('indexeddb://next-words-model');
          console.log(`Modell gespeichert (loss: ${logs.loss})`);
        }
      }
    }
  });

  console.log("Training finished!");

  var bestModel = await tf.loadLayersModel('indexeddb://next-words-model');
  await bestModel.save('downloads://next-words-model');
  console.log("Modell heruntergeladen!");
  /*
    const loadedModel2 = await tf.loadLayersModel('indexeddb://next-words-model');
    await loadedModel2.save('downloads://next-words-model');
    console.log('Modell heruntergeladen!');
  */
}

function downloadTokenizer(tokenizerObj) {
  const blob = new Blob([JSON.stringify(tokenizerObj)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'tokenizer.json';
  a.click();
  URL.revokeObjectURL(url);
}
// Bis loadModel wieder reinkommentieren

async function loadModel() {
  try {
    // Modell laden
    model = await tf.loadLayersModel('./next-words-model.json');
    console.log("Modell geladen!");

    const tokenizerResp = await fetch('tokenizer.json');
    wordIndex = await tokenizerResp.json();
    console.log("Tokenizer geladen!");

    // Invertierten Index bauen
    indexWord = {};
    Object.keys(wordIndex).forEach(word => {
      indexWord[wordIndex[word]] = word;
    });

  } catch (e) {
    console.log("Fehler beim Laden von Modell/Tokenizer: ", e);
  }
}

// Direkt beim Start der Seite laden
//loadModel();



/*
var wordIndex = JSON.parse(localStorage.getItem('tokenizer'));

const indexWord = {};
Object.keys(wordIndex).forEach(word => {
  indexWord[wordIndex[word]] = word;
});
*/
/*
function predictNextWord(model, wordIndex, indexWord, textArray) {

  var sequence = textArray.map(w => wordIndex[w] || 0); // 0 falls unbekanntes Wort

  var inputTensor = tf.tensor2d([sequence]);
  var prediction = model.predict(inputTensor);
  var predictedIndex = prediction.argMax(-1).dataSync()[0];

  inputTensor.dispose();
  prediction.dispose();

  var predictedWord = indexWord[predictedIndex];
  console.log(predictedWord);
  return predictedWord;
}
*/
function predictNextWord(textArray) {
  if (!model || !wordIndex) {
    console.log("Modell oder Tokenizer noch nicht geladen!");
    return null;
  }

  const sequence = textArray.map(w => wordIndex[w] || 0);
  const inputTensor = tf.tensor2d([sequence]);

  const prediction = model.predict(inputTensor);
  const predictedIndex = prediction.argMax(-1).dataSync()[0];

  inputTensor.dispose();
  prediction.dispose();

  const predictedWord = indexWord[predictedIndex];
  console.log(predictedWord);
  return predictedWord;
}

//ZUSTAND
let stopRequested = false;
let autoRunning = false;

function getWords() {
  const text = document.getElementById('userInput').value;
  return text.split(' ').filter(w => w.length > 0);
}

function getLastThreeWords() {
  return getWords().slice(-3);
}

function appendWord(word) {
  const textarea = document.getElementById('userInput');
  const words = getWords();
  words.push(word);
  textarea.value = words.join(' ') + ' ';
  autoResizeTextarea();
}

function setStatus(msg) {
  document.getElementById('status').innerText = msg;
}

function clearPredictions() {
  document.getElementById('predictions').innerHTML = '';
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

//TOP-K VORHERSAGE MIT WAHRSCHEINLICHKEITEN
function predictTopK(textArray, k = 5) {
  if (!model || !wordIndex) {
    setStatus("Modell oder Tokenizer noch nicht geladen!");
    return null;
  }

  return tf.tidy(() => {
    const sequence = textArray.map(w => wordIndex[w] || 0);
    const inputTensor = tf.tensor2d([sequence]);

    const prediction = model.predict(inputTensor); // bereits Softmax, Shape [1, vocabSize]
    const { values, indices } = tf.topk(prediction, k);

    const probs = values.dataSync();
    const idxs = indices.dataSync();

    const results = [];
    for (let i = 0; i < k; i++) {
      results.push({
        word: indexWord[idxs[i]] || '(unbekannt)',
        prob: probs[i]
      });
    }
    return results;
  });
}

//VORHERSAGE BERECHNEN (ohne anzuhängen)
function computePredictions() {
  const lastThree = getLastThreeWords();
  if (lastThree.length < 3) {
    setStatus("Bitte mindestens 3 vollständige Wörter eingeben.");
    clearPredictions();
    return null;
  }
  setStatus("");
  return predictTopK(lastThree, 5);
}

//VORHERSAGEN ALS KLICKBARE BUTTONS ANZEIGEN
function showPredictions(predictions) {
  clearPredictions();
  const container = document.getElementById('predictions');

  predictions.forEach(p => {
    const btn = document.createElement('button');
    btn.textContent = `${p.word} (${(p.prob * 100).toFixed(1)}%)`;
    btn.addEventListener('click', () => {
      appendWord(p.word);
      doVorhersage();
    });
    container.appendChild(btn);
  });
}


function doVorhersage() {
  const predictions = computePredictions();
  if (!predictions) return;
  showPredictions(predictions);
}

document.getElementById('vorhersageBtn').addEventListener('click', doVorhersage);


function doWeiter() {
  const predictions = computePredictions();
  if (!predictions) return;

  const best = predictions[0]; // wahrscheinlichstes Wort
  appendWord(best.word);
  doVorhersage();
}

document.getElementById('weiterBtn').addEventListener('click', doWeiter);


async function doAuto() {
  autoRunning = true;
  stopRequested = false;
  document.getElementById('autoBtn').disabled = true;
  document.getElementById('stoppBtn').disabled = false;

  for (let i = 0; i < 10; i++) {
    if (stopRequested) break;

    const predictions = computePredictions();
    if (!predictions) break;

    const best = predictions[0];
    appendWord(best.word);
    showPredictions(predictions);

    await sleep(400); // kleine Pause, damit man den Fortschritt sieht
  }

  autoRunning = false;
  document.getElementById('autoBtn').disabled = false;
  document.getElementById('stoppBtn').disabled = true;

  // Am Ende die aktuell gültige Vorhersage anzeigen
  doVorhersage();
}

document.getElementById('autoBtn').addEventListener('click', doAuto);


document.getElementById('stoppBtn').addEventListener('click', () => {
  stopRequested = true;
});


document.getElementById('resetBtn').addEventListener('click', () => {
  stopRequested = true; // falls Auto gerade läuft, stoppen
  document.getElementById('userInput').value = '';
  clearPredictions();
  setStatus('');
  autoResizeTextarea();
});


function autoResizeTextarea() {
  const textarea = document.getElementById('userInput');
  console.log("Resize läuft, scrollHeight:", textarea.scrollHeight);
  textarea.style.height = 'auto';
  textarea.style.height = textarea.scrollHeight + 'px';
}

document.getElementById('userInput').addEventListener('input', autoResizeTextarea);



hello();
