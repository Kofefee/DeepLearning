async function hello() {
  var vocabResp = await fetch('dataset-test.txt');
  let text = await vocabResp.text();
  console.log("Start Training!");

  text = text
    .replace(/\n/g, ' ')
    .replace(/\r/g, ' ')
    .replace(/\ufeff/g, '')
    .replace(/"/g, '');

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
  var EPOCHS = 1;

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
loadModel();



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

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('predictBtn').addEventListener('click', () => {
    const text = document.getElementById('userInput').value;

    if (text === "0") {
      console.log("Execution completed....");
      return;
    }

    try {
      const words = text.split(" ").filter(w => w.length > 0);
      const lastThree = words.slice(-3);
      console.log(lastThree);

      if (lastThree.length < 3) {
        console.log("Bitte mindestens 3 Wörter eingeben.");
        return;
      }

      const predicted = predictNextWord(lastThree);
      document.getElementById('result').innerText = "Vorhersage: " + predicted;

    } catch (e) {
      console.log("Error occured: ", e);
    }
  });
});
//hello();
