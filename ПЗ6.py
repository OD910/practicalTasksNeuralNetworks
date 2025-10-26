#вариант 1

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle



def gen_rect(size=50):
    img = np.zeros([size, size])
    x = np.random.randint(0, size)
    y = np.random.randint(0, size)
    w = np.random.randint(size // 10, size // 2)
    h = np.random.randint(size // 10, size // 2)
    img[x:x + w, y:y + h] = 1.0  # Используем float
    return img


def gen_circle(size=50):
    img = np.zeros([size, size])
    x = np.random.randint(0, size)
    y = np.random.randint(0, size)
    r = np.random.randint(size // 10, size // 3)
    for i in range(0, size):
        for j in range(0, size):
            if (i - x) ** 2 + (j - y) ** 2 <= r ** 2:
                img[i, j] = 1.0  # Используем float
    return img


def gen_empty_circle(size=50):
    img = np.zeros([size, size])
    x = np.random.randint(0, size)
    y = np.random.randint(0, size)
    r = np.random.randint(size // 10, size // 3)
    dr = np.random.randint(1, 10) + r
    for i in range(0, size):
        for j in range(0, size):
            if r ** 2 <= (i - x) ** 2 + (j - y) ** 2 <= dr ** 2:
                img[i, j] = 1.0
    return img


def gen_h_line(size=50):
    img = np.zeros([size, size])
    x = np.random.randint(10, size - 10)
    y = np.random.randint(10, size - 10)
    l = np.random.randint(size // 8, size // 2)
    w = 1
    img[x - w:x + w, y - l:y + l] = 1.0
    return img


def gen_v_line(size=50):
    img = np.zeros([size, size])
    x = np.random.randint(10, size - 10)
    y = np.random.randint(10, size - 10)
    l = np.random.randint(size // 8, size // 2)
    w = 1
    img[x - l:x + l, y - w:y + w] = 1.0
    return img


def gen_cross(size=50):
    img = np.zeros([size, size])
    x = np.random.randint(10, size - 10)
    y = np.random.randint(10, size - 10)
    l = np.random.randint(size // 8, size // 5)
    w = 1
    img[x - l:x + l, y - w:y + w] = 1.0
    img[x - w:x + w, y - l:y + l] = 1.0
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


#БЛОК 3: Вспомогательные функции (Визуализация)

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # График точности
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')

    # График потерь
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')

    plt.show()


def show_predictions(data, labels, model, class_names):
    plt.figure(figsize=(10, 10))
    indices = np.random.choice(len(data), 9, replace=False)

    for i, idx in enumerate(indices):
        ax = plt.subplot(3, 3, i + 1)
        img = data[idx]

        # Предсказание
        img_tensor = np.expand_dims(img, axis=0)
        prediction = model.predict(img_tensor, verbose=0)

        predicted_index = int(prediction[0][0] > 0.5)
        predicted_class = class_names[predicted_index]
        true_class = class_names[labels[idx]]

        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f"True: {true_class}\nPred: {predicted_class}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# БЛОК 4: ОСНОВНОЙ СКРИПТ (Предобработка, CNN, Обучение)

if __name__ == "__main__":
    IMG_SIZE = 50
    N_SAMPLES = 2000

    # 1. Загрузка и Предобработка Данных
    print(f"Генерация {N_SAMPLES} изображений...")
    data, labels_str = gen_data(size=N_SAMPLES, img_size=IMG_SIZE)

    # a) Преобразование строковых меток в числа (0, 1)
    encoder = LabelEncoder()
    labels_numeric = encoder.fit_transform(labels_str.ravel())
    class_names = encoder.classes_
    print(f"Классы: {list(class_names)}")

    # b) Добавление канала (N, 50, 50) -> (N, 50, 50, 1)
    data_cnn = np.expand_dims(data, axis=-1)
    print(f"Новая форма данных (с каналом): {data_cnn.shape}")

    # c) Перемешивание данных (КРИТИЧЕСКИ ВАЖНО!)
    X, y = shuffle(data_cnn, labels_numeric, random_state=42)

    # d) Разделение на обучающую и валидационную выборки
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Обучающая выборка: {X_train.shape[0]} изображений")
    print(f"Валидационная выборка: {X_val.shape[0]} изображений")

    #  2. Построение Модели CNN
    print("\nПостроение модели CNN...")

    model = models.Sequential()
    model.add(layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()

    #  3. Компиляция Модели
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # 4. Обучение Модели
    print("\nНачало обучения...")
    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=32,
        validation_data=(X_val, y_val)
    )

    # 5. Оценка и Визуализация
    print("\nОценка модели...")
    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"\nТочность на валидационной выборке: {val_acc * 100:.2f}%")

    plot_history(history)
    show_predictions(X_val, y_val, model, class_names)






#вариант 2


    #  БЛОК 1: Функции генерации фигур

    def gen_rect(size=50):
        img = np.zeros([size, size])
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        w = np.random.randint(size // 10, size // 2)
        h = np.random.randint(size // 10, size // 2)
        img[x:x + w, y:y + h] = 1.0  # Используем float
        return img


    def gen_circle(size=50):
        img = np.zeros([size, size])
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        r = np.random.randint(size // 10, size // 3)
        for i in range(0, size):
            for j in range(0, size):
                if (i - x) ** 2 + (j - y) ** 2 <= r ** 2:
                    img[i, j] = 1.0  # Используем float
        return img


    def gen_empty_circle(size=50):
        img = np.zeros([size, size])
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        r = np.random.randint(size // 10, size // 3)
        dr = np.random.randint(r + 2, r + 10)  # Делаем "кольцо" толщиной от 2 до 10

        for i in range(0, size):
            for j in range(0, size):
                dist_sq = (i - x) ** 2 + (j - y) ** 2
                if r ** 2 <= dist_sq <= dr ** 2:
                    img[i, j] = 1.0
        return img


    def gen_h_line(size=50):
        img = np.zeros([size, size])
        x = np.random.randint(10, size - 10)
        y = np.random.randint(10, size - 10)
        l = np.random.randint(size // 8, size // 2)
        w = 1
        img[x - w:x + w, y - l:y + l] = 1.0
        return img


    def gen_v_line(size=50):
        img = np.zeros([size, size])
        x = np.random.randint(10, size - 10)
        y = np.random.randint(10, size - 10)
        l = np.random.randint(size // 8, size // 2)
        w = 1
        img[x - l:x + l, y - w:y + w] = 1.0
        return img


    def gen_cross(size=50):
        img = np.zeros([size, size])
        x = np.random.randint(10, size - 10)
        y = np.random.randint(10, size - 10)
        l = np.random.randint(size // 8, size // 5)
        w = 1
        img[x - l:x + l, y - w:y + w] = 1.0
        img[x - w:x + w, y - l:y + l] = 1.0
        return img


    #БЛОК 2: Функция генерации датасета (Вариант 2)

    def gen_data(size=500, img_size=50):
        c1 = size // 2
        c2 = size - c1

        label_c1 = np.full([c1, 1], 'Empty')
        # Вызываем gen_empty_circle() напрямую
        data_c1 = np.array([gen_empty_circle(img_size) for i in range(c1)])

        label_c2 = np.full([c2, 1], 'Not Empty')
        # Вызываем gen_circle() напрямую
        data_c2 = np.array([gen_circle(img_size) for i in range(c2)])

        data = np.vstack((data_c1, data_c2))
        label = np.vstack((label_c1, label_c2))

        return data, label


    #  БЛОК 3: Вспомогательные функции (Визуализация)

    def plot_history(history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # График точности
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend(loc='lower right')

        # График потерь
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend(loc='upper right')

        plt.show()


    def show_predictions(data, labels, model, class_names):
        plt.figure(figsize=(10, 10))
        indices = np.random.choice(len(data), 9, replace=False)

        for i, idx in enumerate(indices):
            ax = plt.subplot(3, 3, i + 1)
            img = data[idx]

            # Предсказание
            img_tensor = np.expand_dims(img, axis=0)
            prediction = model.predict(img_tensor, verbose=0)

            predicted_index = int(prediction[0][0] > 0.5)
            predicted_class = class_names[predicted_index]
            true_class = class_names[labels[idx]]

            plt.imshow(img.squeeze(), cmap='gray')
            plt.title(f"True: {true_class}\nPred: {predicted_class}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()


    #  БЛОК 4: ОСНОВНОЙ СКРИПТ (Предобработка, CNN, Обучение)

    if __name__ == "__main__":
        IMG_SIZE = 50
        N_SAMPLES = 2000

        #  1. Загрузка и Предобработка Данных
        print(f"Генерация {N_SAMPLES} изображений (Вариант 2)...")
        data, labels_str = gen_data(size=N_SAMPLES, img_size=IMG_SIZE)

        # a) Преобразование строковых меток в числа (0, 1)
        # 'Empty' -> 0, 'Not Empty' -> 1 (или наоборот, LabelEncoder решит)
        encoder = LabelEncoder()
        labels_numeric = encoder.fit_transform(labels_str.ravel())
        class_names = encoder.classes_
        print(f"Классы: {list(class_names)}")

        # b) Добавление канала (N, 50, 50) -> (N, 50, 50, 1)
        data_cnn = np.expand_dims(data, axis=-1)
        print(f"Новая форма данных (с каналом): {data_cnn.shape}")

        # c) Перемешивание данных (КРИТИЧЕСКИ ВАЖНО!)
        X, y = shuffle(data_cnn, labels_numeric, random_state=42)

        # d) Разделение на обучающую и валидационную выборки
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Обучающая выборка: {X_train.shape[0]} изображений")
        print(f"Валидационная выборка: {X_val.shape[0]} изображений")

        #  2. Построение Модели CNN
        print("\nПостроение модели CNN...")

        model = models.Sequential()
        model.add(layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.summary()

        #  3. Компиляция Модели
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        #  4. Обучение Модели
        print("\nНачало обучения...")
        history = model.fit(
            X_train, y_train,
            epochs=15,
            batch_size=32,
            validation_data=(X_val, y_val)
        )

        #5. Оценка и Визуализация
        print("\nОценка модели...")
        val_loss, val_acc = model.evaluate(X_val, y_val)
        print(f"\nТочность на валидационной выборке: {val_acc * 100:.2f}%")

        plot_history(history)
        show_predictions(X_val, y_val, model, class_names)