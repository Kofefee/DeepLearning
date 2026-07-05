import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import pickle
import numpy as np
import os

file = open('dataset-test.txt', 'r', encoding = "utf8")
lines=[]
for i in file:
  lines.append(i)

data = ""
for i in lines:
  data = ' '.join(lines)
  
  
data.replace('\n', '').replace('\r', '').replace('\ufeff', '').replace('"','').replace('"', '')

data = data.split()
data = ' '.join(data)

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

pickle.dump(tokenizer, open('token.pkl', 'wb'))
sequence_data = tokenizer.texts_to_sequences([data])[0]
sequence = []
for i in range (3, len(sequence_data)):
    words = sequence_data[i-3: i+1]
    sequence.append(words)
    
sequence = np.array(sequence)

x = []
y = []

for i in sequence:
    x.append(i[0:3])
    y.append(i[3])
    
x = np.array(x)
y = np.array(y)
vocabulary_size = len(tokenizer.word_index)+1

y = to_categorical(y,num_classes = vocabulary_size)

model = Sequential()
model.add(Embedding(vocabulary_size, 10))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation="relu"))
model.add(Dense(vocabulary_size, activation="softmax"))

from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint_path = "training_2/cp.keras"
checkpoint_dir = os.path.dirname(checkpoint_path)
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True)
model.compile(loss="categorical_crossentropy", optimizer= Adam(learning_rate=0.01))
model.fit(x, y, epochs=2, batch_size=16, callbacks=[checkpoint])

model = tensorflow.keras.models.load_model('training_2/cp.keras')

model.save('cp.h5')

print("Fertig: cp.h5 wurde gespeichert.")