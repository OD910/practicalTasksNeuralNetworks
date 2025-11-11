import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import Callback
import os
from datetime import datetime


#  БЛОК 1: Функции генерации фигур

def gen_rect(size=50):
    img = np.zeros([size, size])
    x = np.random.randint(0, size)
    y = np.random.randint(0, size)
    w = np.random.randint(size // 10, size // 2)
    h = np.random.randint(size // 10, size // 2)
    img[x:x + w, y:y + h] = 1.0
    return img


def gen_circle(size=50):
    img = np.zeros([size, size])
    x = np.random.randint(0, size)
    y = np.random.randint(0, size)
    r = np.random.randint(size // 10, size // 3)
    for i in range(0, size):
        for j in range(0, size):
            if (i - x) ** 2 + (j - y) ** 2 <= r ** 2:
                img[i, j] = 1.0
    return img



#  БЛОК 2: Функция генерации датасета (Вариант 1)

def gen_data(size=500, img_size=50):
    c1 = size // 2
    c2 = size - c1
    label_c1 = np.full([c1, 1], 'Square')
    data_c1 = np.array([gen_rect(img_size) for i in range(c1)])
    label_c2 = np.full([c2, 1], 'Circle')
    data_c2 = np.array([gen_circle(img_size) for i in range(c2)])
    data = np.vstack((data_c1, data_c2))
    label = np.vstack((label_c1, label_c2))
    return data, label



#  БЛОК 3: новый CALLBACK (Вариант 1)



class Top3ModelCheckpoint(Callback):


    def __init__(self, user_prefix):
        super(Top3ModelCheckpoint, self).__init__()
        self.user_prefix = user_prefix
        # Храним кортежи (val_loss, filepath)
        self.top_models = [(np.inf, None), (np.inf, None), (np.inf, None)]
        self.date_str = datetime.now().strftime('%Y-%m-%d')
        self.temp_count = 0

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        if current_loss is None:
            return

        # Проверяем, лучше ли текущая модель, чем худшая из топ-3
        if current_loss < self.top_models[-1][0]:
            # Нашли новую лучшую модель ---

            # 1. Сохраняем новую модель с временным именем
            self.temp_count += 1
            temp_path = f"{self.date_str}_{self.user_prefix}_temp_{self.temp_count}.h5"
            self.model.save(temp_path)

            # 2. Добавляем ее в список и сортируем
            self.top_models.append((current_loss, temp_path))
            self.top_models.sort(key=lambda x: x[0])

            # 3. Убираем худшую (4-ю) модель и удаляем ее файл
            model_to_remove = self.top_models.pop()
            if model_to_remove[1] and os.path.exists(model_to_remove[1]):
                os.remove(model_to_remove[1])

            print(f"\nНовая модель (Loss: {current_loss:.4f}) вошла в Топ-3.")

            # 4. Переименовываем 3 лучшие модели в правильные имена

            for i in reversed(range(len(self.top_models))):
                rank = i + 1
                loss, path = self.top_models[i]
                new_path = f"{self.date_str}_{self.user_prefix}_{rank}.h5"

                if path and path != new_path:
                    if os.path.exists(path):
                        os.rename(path, new_path)
                    # Обновляем путь в нашем списке
                    self.top_models[i] = (loss, new_path)

            print("Топ-3 обновлен:")
            for i, (loss, path) in enumerate(self.top_models):
                if path:  # Печатаем только те, что уже сохранены
                    print(f"  Rank {i + 1}: {path} (Loss: {loss:.4f})")

    def on_train_end(self, logs=None):
        # Очистка временных файлов
        for file in os.listdir('.'):
            if "temp" in file and self.user_prefix in file:
                try:
                    os.remove(file)
                except OSError:
                    pass  # На случай, если файл уже был переименован



#  БЛОК 4: Вспомогательные функции (Визуализация)

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend(loc='lower right')
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.legend(loc='upper right')
    plt.show()


#БЛОК 5: ОСНОВНОЙ СКРИПТ (Предобработка, CNN, Обучение)

if __name__ == "__main__":

    IMG_SIZE = 50
    N_SAMPLES = 2000

    # 1. Предобработка Данных
    print(f"Генерация {N_SAMPLES} изображений (Вариант 1)...")
    data, labels_str = gen_data(size=N_SAMPLES, img_size=IMG_SIZE)

    encoder = LabelEncoder()
    labels_numeric = encoder.fit_transform(labels_str.ravel())
    class_names = encoder.classes_
    print(f"Классы: {list(class_names)}")

    data_cnn = np.expand_dims(data, axis=-1)
    X, y = shuffle(data_cnn, labels_numeric, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    #  2. Построение Модели CNN
    print("\nПостроение модели CNN...")
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    print("\nНачало обучения...")

    # Задаем префикс для имени файла
    user_prefix = "CNN_Variant1"
    top_3_callback = Top3ModelCheckpoint(user_prefix=user_prefix)

    history = model.fit(
        X_train, y_train,
        epochs=15,  # Обучаем 15 эпох
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[top_3_callback]  # Передаем наш callback в обучение
    )

    print("\nОценка лучшей модели...")
    # Загружаем САМУЮ ЛУЧШУЮ модель (с рангом 1)
    best_model_path = f"{datetime.now().strftime('%Y-%m-%d')}_{user_prefix}_1.h5"
    if os.path.exists(best_model_path):
        best_model = models.load_model(best_model_path)
        val_loss, val_acc = best_model.evaluate(X_val, y_val)
        print(f"\nТочность ЛУЧШЕЙ модели (из файла {best_model_path}) на валидации: {val_acc * 100:.2f}%")
    else:
        print("Не удалось найти сохраненную лучшую модель.")

    plot_history(history)






# ВАРИАНТ2



    #  БЛОК 1: Функции генерации фигур

    def gen_circle(size=50):
        img = np.zeros([size, size])
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        r = np.random.randint(size // 10, size // 3)
        for i in range(0, size):
            for j in range(0, size):
                if (i - x) ** 2 + (j - y) ** 2 <= r ** 2:
                    img[i, j] = 1.0
        return img


    def gen_empty_circle(size=50):
        img = np.zeros([size, size])
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        r = np.random.randint(size // 10, size // 3)
        # Гарантируем, что кольцо будет иметь видимую толщину
        dr = np.random.randint(r + 2, r + 10)

        for i in range(0, size):
            for j in range(0, size):
                dist_sq = (i - x) ** 2 + (j - y) ** 2
                if r ** 2 <= dist_sq <= dr ** 2:
                    img[i, j] = 1.0
        return img



    # БЛОК 2: Функция генерации датасета (Вариант 2)

    def gen_data(size=500, img_size=50):
        c1 = size // 2
        c2 = size - c1
        label_c1 = np.full([c1, 1], 'Empty')
        data_c1 = np.array([gen_empty_circle(img_size) for i in range(c1)])
        label_c2 = np.full([c2, 1], 'Not Empty')
        data_c2 = np.array([gen_circle(img_size) for i in range(c2)])
        data = np.vstack((data_c1, data_c2))
        label = np.vstack((label_c1, label_c2))
        return data, label

    #блок3

    class FeatureMapVisualizer(Callback):
        # Сохраняет ядра (карты признаков) сверточных слоев в заданные эпохи.

        def __init__(self, epochs_to_save, conv_layer_names):
            super(FeatureMapVisualizer, self).__init__()
            self.epochs_to_save = set(epochs_to_save)  # Используем set для O(1)
            self.conv_layer_names = conv_layer_names
            # Создаем папку для сохранения изображений
            self.output_dir = "feature_maps"
            os.makedirs(self.output_dir, exist_ok=True)

        def on_epoch_end(self, epoch, logs=None):
            current_epoch = epoch + 1

            if current_epoch in self.epochs_to_save:
                print(f"\n--- Сохранение ядер (карт признаков) для эпохи {current_epoch} ---")

                for layer_name in self.conv_layer_names:
                    layer = self.model.get_layer(layer_name)
                    # Получаем веса (ядра) и смещения слоя
                    weights, biases = layer.get_weights()

                    # Форма весов: (height, width, input_channels, output_channels/filters)
                    n_filters = weights.shape[3]

                    for i in range(n_filters):
                        # Так как у нас 1 входной канал (ЧБ), берем [:, :, 0, i]
                        kernel = weights[:, :, 0, i]

                        kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())

                        plt.figure()
                        plt.imshow(kernel, cmap='gray')
                        plt.title(f'Слой: {layer_name}, Ядро: {i + 1}, Эпоха: {current_epoch}')
                        plt.axis('off')

                        # <номер слоя>_<номер ядра в слое>_<номер эпохи>
                        filename = f"{layer_name}_{i + 1}_{current_epoch}.png"
                        filepath = os.path.join(self.output_dir, filename)
                        plt.savefig(filepath)
                        plt.close()
                print(f"Все ядра для эпохи {current_epoch} сохранены в папку '{self.output_dir}'")


    #  БЛОК 4: Вспомогательные функции

    def plot_history(history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.legend(loc='lower right')
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.legend(loc='upper right')
        plt.show()


    # БЛОК 5: ОСНОВНОЙ СКРИПТ (Предобработка, CNN, Обучение)


    if __name__ == "__main__":
        IMG_SIZE = 50
        N_SAMPLES = 2000

        # 1. Предобработка Данных
        print(f"Генерация {N_SAMPLES} изображений (Вариант 2)...")
        data, labels_str = gen_data(size=N_SAMPLES, img_size=IMG_SIZE)

        encoder = LabelEncoder()
        labels_numeric = encoder.fit_transform(labels_str.ravel())
        class_names = encoder.classes_
        print(f"Классы: {list(class_names)}")

        data_cnn = np.expand_dims(data, axis=-1)
        X, y = shuffle(data_cnn, labels_numeric, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        #  2. Построение Модели CNN
        print("\nПостроение модели CNN...")

        model = models.Sequential([
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
            # !!! МЫ ДОБАВИЛИ ИМЕНА СЛОЯМ, чтобы callback мог их найти !!!
            layers.Conv2D(32, (3, 3), activation='relu', name='conv_1'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', name='conv_2'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', name='conv_3'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        model.summary()

        print("\nНачало обучения...")

        # Задаем эпохи, на которых хотим сохранить ядра
        epochs_to_save_maps = [1, 5, 10, 15]
        # Задаем имена слоев, которые хотим визуализировать
        layers_to_visualize = ['conv_1', 'conv_2']

        fm_callback = FeatureMapVisualizer(
            epochs_to_save=epochs_to_save_maps,
            conv_layer_names=layers_to_visualize
        )

        history = model.fit(
            X_train, y_train,
            epochs=15,  # Обучаем 15 эпох
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[fm_callback]  # Передаем наш callback в обучение
        )

        # 4. Оценка
        print("\nОценка модели...")
        val_loss, val_acc = model.evaluate(X_val, y_val)
        print(f"\nТочность на валидационной выборке: {val_acc * 100:.2f}%")

        plot_history(history)