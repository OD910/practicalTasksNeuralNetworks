import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam


def logical_operation(a, b, c):
    # Вариант 3: (a and b) or c
    return (a and b) or c


# Создание полного набора данных для 3 входов (2^3 = 8 комбинаций)
# Входные данные (X)
data_input = np.array([[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]], dtype=np.float32)

# Ожидаемые выходные данные (y)
labels = np.array([logical_operation(i[0], i[1], i[2]) for i in data_input], dtype=np.float32).reshape(-1, 1)

# Создаем простую последовательную модель
model = models.Sequential([
    # Входной слой (3 нейрона по числу входов) и один скрытый слой с 4 нейронами
    layers.Dense(8, activation='relu', input_shape=(3,)),
    # Выходной слой с 1 нейроном и сигмоидной активацией для бинарной классификации
    layers.Dense(1, activation='sigmoid')
])

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=0.01),
              loss='binary_crossentropy',
              metrics=['accuracy'])


def simulate_element_wise(data, weights_list):


    # Веса: [W1, b1, W2, b2, ...]
    W1, b1, W2, b2 = weights_list

    final_outputs = []
    # Проходим по каждому примеру в данных
    for sample in data:
        # Расчет для скрытого слоя
        hidden_layer_outputs = []
        # Проходим по каждому нейрону скрытого слоя
        for j in range(W1.shape[1]):  # W1.shape[1] - количество нейронов
            dot_product = 0
            # Считаем взвешенную сумму входов
            for i in range(len(sample)):
                dot_product += sample[i] * W1[i, j]
            # Добавляем смещение (bias)
            z = dot_product + b1[j]
            # Применяем активацию ReLU
            activation = max(0, z)
            hidden_layer_outputs.append(activation)

        #  Расчет для выходного слоя
        dot_product_out = 0
        # Считаем взвешенную сумму выходов скрытого слоя
        for j in range(len(hidden_layer_outputs)):
            dot_product_out += hidden_layer_outputs[j] * W2[j, 0]
        # Добавляем смещение
        z_out = dot_product_out + b2[0]
        # Применяем активацию Sigmoid
        final_output = 1 / (1 + np.exp(-z_out))
        final_outputs.append(final_output)

    return np.array(final_outputs).reshape(-1, 1)


def simulate_numpy(data, weights_list):

    W1, b1, W2, b2 = weights_list

    # Расчет для скрытого слоя
    z1 = data @ W1 + b1
    a1 = np.maximum(0, z1)  # активация ReLU

    # Расчет для выходного слоя
    z2 = a1 @ W2 + b2
    output = 1 / (1 + np.exp(-z2))  # активация Sigmoid

    return output


print("=" * 25 + " ДО ОБУЧЕНИЯ " + "=" * 25)

# 1. Получаем веса из необученной модели
weights_before_training = model.get_weights()

# 2. проганяем датасет
keras_pred_before = model.predict(data_input)
sim_elem_before = simulate_element_wise(data_input, weights_before_training)
sim_numpy_before = simulate_numpy(data_input, weights_before_training)

# 3. Сравниваем результаты
print("Входные данные -> Keras Model -> Element-wise -> NumPy")
for i in range(len(data_input)):
    print(
        f"{data_input[i]} -> {keras_pred_before[i][0]:.5f} -> {sim_elem_before[i][0]:.5f} -> {sim_numpy_before[i][0]:.5f}")

# Обучаем модель
print("\n...Обучение модели...\n")
model.fit(data_input, labels, epochs=200, verbose=0)

print("=" * 25 + " ПОСЛЕ ОБУЧЕНИЯ " + "=" * 25)

# 4. Получаем веса из ОБУЧЕННОЙ модели
weights_after_training = model.get_weights()

# 5. Прогоняем датасет через 3 варианта
keras_pred_after = model.predict(data_input)
sim_elem_after = simulate_element_wise(data_input, weights_after_training)
sim_numpy_after = simulate_numpy(data_input, weights_after_training)

# 6. Сравниваем результаты
print("Входные данные -> Keras Model -> Element-wise -> NumPy")
for i in range(len(data_input)):
    print(
        f"{data_input[i]} -> {keras_pred_after[i][0]:.5f} -> {sim_elem_after[i][0]:.5f} -> {sim_numpy_after[i][0]:.5f}")

# Финальная проверка точности
loss, accuracy = model.evaluate(data_input, labels, verbose=0)
print(f"\nИтоговая точность обученной модели: {accuracy * 100:.2f}%")