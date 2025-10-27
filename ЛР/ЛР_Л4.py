import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image  # <--- ВОТ НОВЫЙ ИМПОРТ

# 1. Загрузка и подготовка данных
print("Загрузка данных MNIST...")
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Меняем форму для CNN (добавляем 1 канал цвета)
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Нормализация
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Кодируем метки
train_labels_cat = to_categorical(train_labels)
test_labels_cat = to_categorical(test_labels)


#2. Функция создания CNN
def create_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))  # 10 цифр

    # Компилируем модель
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# 3. Загрузка или Обучение Модели
MODEL_FILENAME = "mnist_model.h5"  # Имя файла модели

if os.path.exists(MODEL_FILENAME):
    #  Если файл есть - загружаем
    print(f"\nЗагрузка существующей модели из {MODEL_FILENAME}...")
    model = load_model(MODEL_FILENAME)
    print("Модель успешно загружена.")

else:
    #Если файла нет - обучаем и сохраняем
    print(f"\nФайл модели {MODEL_FILENAME} не найден. Запуск нового обучения...")
    model = create_cnn_model()

    # Обучаем
    print("Обучение...")
    history = model.fit(train_images,
                        train_labels_cat,
                        epochs=5,
                        batch_size=128,
                        validation_split=0.1)

    # Сохраняем
    model.save(MODEL_FILENAME)
    print(f"Обучение завершено. Модель сохранена в {MODEL_FILENAME}.")

#  4. Оценка модели
print("\nОценка точности модели на тестовых данных...")
test_loss, test_acc = model.evaluate(test_images, test_labels_cat)
print(f"Точность на тестовых данных: {test_acc * 100:.2f}%")


#  5. Функция для классификации
def classify_custom_image(model_to_use, image_path):

    if not os.path.exists(image_path):
        # Если файла нет, просто сообщаем и пропускаем
        print(f"\n--- Файл {image_path} не найден, пропуск. ---")
        return

    try:
        # 1. Открываем изображение с помощью Pillow (PIL)
        img = Image.open(image_path).convert('RGBA')

        # 2. Создаем чистый белый фон
        bg = Image.new('RGB', img.size, (255, 255, 255))

        # 3. Наклеиваем изображение (img) на белый фон (bg)
        # используя альфа-канал (img.split()[3]) как маску
        bg.paste(img, mask=img.split()[3])

        # 4. Конвертируем в ЧБ (L - Luminance) и меняем размер
        bg = bg.convert('L')
        bg = bg.resize((28, 28))

        # 5. Преобразуем в массив numpy
        img_array = img_to_array(bg)

        # 6. Инвертируем
        img_array = 255.0 - img_array

        # 7. Нормализация и изменение формы
        img_array = (img_array / 255.0).reshape((1, 28, 28, 1))

        # 8. Предсказание
        prediction = model_to_use.predict(img_array)
        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        print(f"\n--- Классификация файла: {image_path} ---")
        print(f"Предсказанная цифра: {digit}")
        print(f"Уверенность: {confidence:.2f}%")

        plt.imshow(img_array.reshape(28, 28), cmap=plt.cm.binary)
        plt.title(f"Предсказание: {digit}")
        plt.show()

    except Exception as e:
        print(f"Произошла ошибка при обработке изображения {image_path}: {e}")


#  6. Запуск классификации для всех 10 цифр
print("\n Запуск классификации 10 пользовательских изображений")

# Создаем список имен файлов от 'digit_0.png' до 'digit_9.png'
filenames_to_test = []
for i in range(10):  # Цикл от 0 до 9
    filenames_to_test.append(f"digit_{i}.png")

# Перебираем все файлы в списке и вызываем функцию
for file in filenames_to_test:
    classify_custom_image(model, file)

print(" Проверка всех файлов завершена. ")