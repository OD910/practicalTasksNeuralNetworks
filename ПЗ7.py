#вариант 1

import numpy as np
import random
import math
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

def gen_sequence(seq_len=1000):
    seq = [math.sin(i/5)/2 + math.cos(i/3)/2 + random.normalvariate(0, 0.04) for i in range(seq_len)]
    return np.array(seq)

def gen_data_from_sequence(seq_len=1200, lookback=20):
    seq = gen_sequence(seq_len)
    past = np.array([[[seq[j]] for j in range(i, i + lookback)] for i in range(len(seq) - lookback)])
    future = np.array([[seq[i]] for i in range(lookback, len(seq))])
    return past, future

lookback = 20
data, res = gen_data_from_sequence(seq_len=1500, lookback=lookback)

dataset_size = len(data)
train_size = int(dataset_size * 0.7)
val_size = int(dataset_size * 0.15)

train_data, train_res = data[:train_size], res[:train_size]
val_data, val_res = data[train_size:train_size+val_size], res[train_size:train_size+val_size]
test_data, test_res = data[train_size+val_size:], res[train_size+val_size:]

print(f"Train shape: {train_data.shape}, Val shape: {val_data.shape}, Test shape: {test_data.shape}")

model = Sequential()

model.add(layers.LSTM(32, input_shape=(lookback, 1), return_sequences=True))
model.add(layers.Dropout(0.2))
model.add(layers.GRU(32))
model.add(layers.Dense(1))

model.compile(optimizer='nadam', loss='mse')

history = model.fit(train_data, train_res, epochs=30, batch_size=32, validation_data=(val_data, val_res), verbose=1)


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('График обучения (Loss)')
plt.xlabel('Epochs')
plt.legend()

predicted_res = model.predict(test_data)

plt.subplot(1, 2, 2)
limit = 100
plt.plot(test_res[:limit], label='Реальные данные', color='blue')
plt.plot(predicted_res[:limit], label='Предсказание сети', color='red', linestyle='--')
plt.title('Предсказание vs Реальность (Тест)')
plt.legend()
plt.show()



#вариант 2


def gen_sequence(seq_len=1000):
    seq = [math.sin(i/5)/2 + math.cos(i/3)/2 + random.normalvariate(0, 0.04) for i in range(seq_len)]
    return np.array(seq)

def gen_data_from_sequence(seq_len=1200, lookback=20):
    seq = gen_sequence(seq_len)
    past = np.array([[[seq[j]] for j in range(i, i + lookback)] for i in range(len(seq) - lookback)])
    future = np.array([[seq[i]] for i in range(lookback, len(seq))])
    return past, future

lookback = 20
data, res = gen_data_from_sequence(seq_len=1500, lookback=lookback)

dataset_size = len(data)
train_size = int(dataset_size * 0.7)
val_size = int(dataset_size * 0.15)

train_data, train_res = data[:train_size], res[:train_size]
val_data, val_res = data[train_size:train_size+val_size], res[train_size:train_size+val_size]
test_data, test_res = data[train_size+val_size:], res[train_size+val_size:]

print(f"Train shape: {train_data.shape}, Val shape: {val_data.shape}, Test shape: {test_data.shape}")

model = Sequential()
model.add(layers.LSTM(32, input_shape=(lookback, 1), return_sequences=True))
model.add(layers.Dropout(0.2))
model.add(layers.GRU(32))
model.add(layers.Dense(1))

model.compile(optimizer='nadam', loss='mse')

history = model.fit(train_data, train_res, epochs=30, batch_size=32, validation_data=(val_data, val_res), verbose=1)


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('График обучения (Loss)')
plt.xlabel('Epochs')
plt.legend()

predicted_res = model.predict(test_data)

plt.subplot(1, 2, 2)
limit = 100
plt.plot(test_res[:limit], label='Реальные данные', color='blue')
plt.plot(predicted_res[:limit], label='Предсказание сети', color='red', linestyle='--')
plt.title('Предсказание vs Реальность (Тест)')
plt.legend()
plt.show()