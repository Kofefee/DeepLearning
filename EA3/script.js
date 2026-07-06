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
  var wordIndex = {};
  sortedWords.forEach((w, i) => { wordIndex[w] = i + 1; });
  localStorage.setItem('tokenizer', JSON.stringify(wordIndex));

  var sequenceData = words.map(w => wordIndex[w]);
  var vocabularySize = Object.keys(wordIndex).length + 1;

  var sequence = [];
  for (let i = 3; i < sequenceData.length; i++) {
    sequence.push(sequenceData.slice(i - 3, i + 1));
  }

  console.log("Vocab size:", vocabularySize);
  console.log("Anzahl Samples:", xArr.length);

  var xArr = sequence.map(s => s.slice(0, 3));
  var yArr = sequence.map(s => s[3]);

  var x = tf.tensor2d(xArr);
  var y = tf.tensor1d(yArr, 'int32');

  var SEQ_LEN = 3;
  var EMBED_DIM = 64;
  var LSTM_UNITS = 100;
  var LEARNING_RATE = 0.01;
  var BATCH_SIZE = 32;
  var EPOCHS = 1;

  var model = tf.sequential({
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

  model.compile({
    optimizer: tf.train.adam(LEARNING_RATE),
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy']
  });

  model.summary();

  let bestLoss = Infinity;
  await model.fit(x, y, {
    epochs: EPOCHS,
    batchSize: BATCH_SIZE,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
        if (logs.loss < bestLoss) {
          bestLoss = logs.loss;
          await model.save('indexeddb://next-words-model');
          console.log(`Modell gespeichert (loss: ${logs.loss})`);
        }
      }
    }
  });

  console.log("Training finished!");
  var loadedModel = await tf.loadLayersModel('indexeddb://next-words-model');
}

hello();


/*
const CONFIG = {
  SEQ_LEN: 5,          // Länge der Eingabesequenz (Kontextfenster)
  EMBED_DIM: 64,       // Dimension der Embedding-Schicht
  LSTM_UNITS: 100,     // Units pro LSTM-Layer (Vorgabe: 100)
  LEARNING_RATE: 0.01, // Vorgabe: 0.01 oder 0.001
  BATCH_SIZE: 32,      // Vorgabe: 32
  EPOCHS: 1           // zum Ausprobieren, Loss beobachten
};
*/
/*
async function hello() {
  //onst { x, y, vocabularySize } = await hello();
  var vocabResp = await fetch('dataset-test.txt');
  var text = await vocabResp.text();
  console.log("Start Training!");

  text = text
    .replace(/\n/g, ' ')
    .replace(/\r/g, ' ')
    .replace(/\ufeff/g, '')
    .replace(/"/g, '');

  text = text.split(/\s+/).filter(w => w.length > 0).join(' ');

  var words = text.split(' ');
  var wordCounts = {};
  words.forEach(w => {
    wordCounts[w] = (wordCounts[w] || 0) + 1;
  });

  var sortedWords = Object.keys(wordCounts).sort(
    (a, b) => wordCounts[b] - wordCounts[a]
  );

  var wordIndex = {};
  sortedWords.forEach((w, i) => {
    wordIndex[w] = i + 1;
  });

  localStorage.setItem('tokenizer', JSON.stringify(wordIndex));

  var sequenceData = words.map(w => wordIndex[w]);
  var vocabularySize = Object.keys(wordIndex).length + 1;

  var sequence = [];
  for (let i = 3; i < sequenceData.length; i++) {
    sequence.push(sequenceData.slice(i - 3, i + 1));
  }
  var xArr = sequence.map(s => s.slice(0, 3));
  var yArr = sequence.map(s => s[3]);

  var x = tf.tensor2d(xArr);
  var y = tf.oneHot(tf.tensor1d(yArr, 'int32'), vocabularySize);
  //return { x, y, vocabularySize, wordIndex };

  var model = tf.sequential({
    layers: [
      tf.layers.lstm({
        units: 50,
        activation: 'relu',
        returnSequences: true,
        inputShape: [CONFIG.SEQ_LEN, 5]
      }),
      tf.layers.lstm({ units: 30, activation: 'relu' }),
      tf.layers.dense({ units: 1, activation: 'sigmoid' }) // oder softmax + vocabSize
    ]
  });

  model.compile({
    optimizer: tf.train.adam(CONFIG.LEARNING_RATE),
    loss: 'binaryCrossentropy', // oder categoricalCrossentropy
    metrics: ['accuracy']
  });

  model.summary();

  let bestLoss = Infinity;
  await model.fit(x, y, {
    epochs: CONFIG.EPOCHS,
    batchSize: CONFIG.BATCH_SIZE,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
        if (logs.loss < bestLoss) {
          bestLoss = logs.loss;
          await model.save('indexeddb://next-words-model');
          console.log(`Modell gespeichert (loss: ${logs.loss})`);
        }
      }
    }
  });

  console.log("Training finished!");
  var loadedModel = await tf.loadLayersModel('indexeddb://next-words-model');



  /*
  async function hello() {
    const vocabResp = await fetch('dataset-test.txt');
    console.log("Start Training!");
  
  var model = tf.sequential({
      layers: [
        tf.layers.lstm({
          units: 50,
          activation: 'relu',
          returnSequences: true,
          inputShape: [10, 5]
        }),
        tf.layers.lstm({
          units: 30,
          activation: 'relu'
        }),
        tf.layers.dense({
          units: 1,
          activation: 'softmax'
        })
      ]
    });
  
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  model.summary()
  
  console.log("Training finished!");
  console.log(model.summary);
  
  let bestLoss = Infinity;
  
  await model.fit(x, y, {
    epochs: 2,
    batchSize: 16,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        //console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
        if (logs.loss < bestLoss) {
          bestLoss = logs.loss;
          await model.save('indexeddb://next-words-model');
          console.log(`Modell gespeichert (verbesserter loss: ${logs.loss})`);
        }
      }
    }
  });
  console.log("Checkpoint erreicht!");
  
  
  tf.loadLayersModel('indexeddb://next-words-model');
  
  };
  /*
    // Export Modell + Vokabular
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
  }
  */
/*
  hello();
} */