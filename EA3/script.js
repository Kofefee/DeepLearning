const CONFIG = {
  SEQ_LEN: 5,          // Länge der Eingabesequenz (Kontextfenster)
  EMBED_DIM: 64,       // Dimension der Embedding-Schicht
  LSTM_UNITS: 100,     // Units pro LSTM-Layer (Vorgabe: 100)
  LEARNING_RATE: 0.01, // Vorgabe: 0.01 oder 0.001
  BATCH_SIZE: 32,      // Vorgabe: 32
  EPOCHS: 1           // zum Ausprobieren, Loss beobachten
};

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

/*
model.compile({
  loss: 'categoricalCrossentropy',
  optimizer: tf.train.adam(0.01)
});
*/
let bestLoss = Infinity;

await model.fit(x, y, {
  epochs: 2,
  batchSize: 16,
  callbacks: {
    onEpochEnd: async (epoch, logs) => {
      console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
      if (logs.loss < bestLoss) {
        bestLoss = logs.loss;
        await model.save('indexeddb://next-words-model');
        console.log(`Modell gespeichert (verbesserter loss: ${logs.loss})`);
      }
    }
  }
});

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
};
hello();