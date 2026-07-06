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


async function hello() {
  const vocabResp = await fetch('dataset-test.txt');



  // Training
  stopTrain = false;
  lossHistory = []; accHistory = [];
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
    }
  });

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

hello();