import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.models import load_model
import os
import sys

# введите последние 2 цифры ИИН
IIN_LAST_TWO_DIGITS = 36

# вариант
VARIANT = 1

TARGET_COLUMN_NUMBER = (IIN_LAST_TWO_DIGITS % 7) + 1
TARGET_COLUMN_INDEX = TARGET_COLUMN_NUMBER - 1

# настройки
DATA_SIZE = 5000
INPUT_DIM = 6  # 6 признаков на вход
CODING_DIM = 3
FILE_PREFIX = f'pr{VARIANT}_{TARGET_COLUMN_NUMBER}_'

print(f"Запуск скрипта")
print(f"Вариант: {VARIANT}")
print(f"Последние 2 цифры ИИН: {IIN_LAST_TWO_DIGITS}")
print(f"Цель регрессии (({IIN_LAST_TWO_DIGITS} % 7) + 1): Признак №{TARGET_COLUMN_NUMBER}")

os.makedirs('results', exist_ok=True)
FILE_PREFIX = os.path.join('results', FILE_PREFIX)

#генерация датасета

def generate_dataset(size, variant):

    #генерирует набор данных в соответствии с вариантом.

    print(f"\n[Шаг 1] Генерация {size} записей для Варианта {variant}...")

    if variant == 4:
        X = np.random.normal(0, 10, size)
        e = np.random.normal(0, 0.3, size)

        f1 = np.cos(X) + e
        f2 = -X + e
        f3 = np.sin(X) * X + e
        f4 = np.sqrt(np.abs(X)) + e
        f5 = X ** 2 + e
        f6 = -np.abs(X) + 4
        f7 = X - (X ** 2) / 5 + e

    elif variant == 1:
        X = np.random.normal(3, 10, size)
        e = np.random.normal(0, 0.3, size)

        f1 = X ** 2 + e
        f2 = np.sin(X / 2) + e
        f3 = np.cos(2 * X) + e
        f4 = X - 3 + e
        f5 = -X + e
        f6 = np.abs(X) + e
        f7 = (X ** 3) / 4 + e

    elif variant == 2:
        X = np.random.normal(-5, 10, size)
        e = np.random.normal(0, 0.3, size)

        f1 = -X ** 3 + e
        f2 = np.log(np.abs(X) + 1e-6) + e  # +1e-6 для избежания log(0)
        f3 = np.sin(3 * X) + e
        f4 = np.exp(X / 10.0) + e  # делим, чтобы избежать переполнения
        f5 = X + 4 + e
        f6 = -X + np.sqrt(np.abs(X)) + e
        f7 = X + e

    elif variant == 3:
        X = np.random.normal(0, 10, size)
        e = np.random.normal(0, 0.3, size)

        f1 = X ** 2 + X + e
        f2 = np.abs(X) + e
        f3 = np.sin(X - np.pi / 4) + e
        f4 = np.log10(np.abs(X) + 1e-6) + e  # +1e-6 для избежания log10(0)
        f5 = -X ** 3 + e
        f6 = -X / 4 + e
        f7 = -X + e

    else:
        print(f"Ошибка: Вариант {variant} не реализован.")
        sys.exit()

    # собираем данные
    columns = [f'feature_{i + 1}' for i in range(7)]
    data = np.stack([f1, f2, f3, f4, f5, f6, f7], axis=1)
    df = pd.DataFrame(data, columns=columns)

    # сохраняем в CSV
    output_path = FILE_PREFIX + 'dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"Датасет сохранен в: {output_path}")
    return df


# 3. построение модели

def build_model(input_dim, coding_dim):


    print(f"\n[Шаг 2] Построение модели...")

    #  общий вход
    input_layer = Input(shape=(input_dim,), name='input_data')

    # кодировщик
    encoded = layers.Dense(12, activation='relu')(input_layer)
    encoded = layers.Dense(8, activation='relu')(encoded)
    bottleneck = layers.Dense(coding_dim, activation='relu', name='bottleneck')(encoded)

    #  декодировщик
    decoded = layers.Dense(8, activation='relu', name='dec_1')(bottleneck)
    decoded = layers.Dense(12, activation='relu', name='dec_2')(decoded)
    decoded_output = layers.Dense(input_dim, activation='linear', name='decoded_out')(decoded)

    # шолова регрессии
    reg = layers.Dense(8, activation='relu', name='reg_1')(bottleneck)
    regression_output = layers.Dense(1, activation='linear', name='regression_out')(reg)

    #  сборка полной модели
    full_model = Model(
        inputs=input_layer,
        outputs=[decoded_output, regression_output]
    )

    # компиляция
    full_model.compile(
        optimizer='adam',
        loss={'decoded_out': 'mse', 'regression_out': 'mse'},
        loss_weights={'decoded_out': 0.5, 'regression_out': 1.0}
    )

    full_model.summary()
    return full_model



# 4. разделение модели


def split_model(full_model, coding_dim):
    print("\n[Шаг 4] Разделение обученной модели на 3 части...")

    # 1. модель Кодировщика
    encoder_model = Model(
        inputs=full_model.input,
        outputs=full_model.get_layer('bottleneck').output
    )
    encoder_model.save(FILE_PREFIX + 'encoder_model.h5')

    # 2. модель Декодировщика
    coded_input = Input(shape=(coding_dim,), name='coded_input')
    d_l1 = full_model.get_layer('dec_1')(coded_input)
    d_l2 = full_model.get_layer('dec_2')(d_l1)
    d_out = full_model.get_layer('decoded_out')(d_l2)
    decoder_model = Model(inputs=coded_input, outputs=d_out)
    decoder_model.save(FILE_PREFIX + 'decoder_model.h5')

    # 3. модель Регрессии (полный путь от входа до регрессии)
    regression_model = Model(
        inputs=full_model.input,
        outputs=full_model.get_layer('regression_out').output
    )
    regression_model.save(FILE_PREFIX + 'regression_model.h5')

    print(f"Модели сохранены в 'results/':")
    print(f"  {FILE_PREFIX}encoder_model.h5")
    print(f"  {FILE_PREFIX}decoder_model.h5")
    print(f"  {FILE_PREFIX}regression_model.h5")

    return encoder_model, decoder_model, regression_model


# 5. основной скрипт

if __name__ == "__main__":
    #  Шаг 1: Генерируем и сохраняем данные
    df = generate_dataset(DATA_SIZE, VARIANT)

    #  Шаг 2: Подготовка данных к обучению
    target_col_name = f'feature_{TARGET_COLUMN_NUMBER}'
    y_reg = df[target_col_name]

    # Входные данные (все, КРОМЕ целевой)
    X = df.drop(columns=[target_col_name])


    y_ae = X

    # Шаг 3: Строим и обучаем модель
    full_model = build_model(input_dim=INPUT_DIM, coding_dim=CODING_DIM)

    print("\n[Шаг 3] Обучение составной модели...")
    full_model.fit(
        X,
        [y_ae, y_reg],
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    #  Шаг 4: Разделяем модель
    encoder_model, decoder_model, regression_model = split_model(full_model, CODING_DIM)

    #  Шаг 5: Генерируем и сохраняем результаты
    print("\n[Шаг 5] Генерация и сохранение CSV результатов...")

    # 1. Закодированные данные
    coded_data = encoder_model.predict(X)
    coded_df = pd.DataFrame(coded_data, columns=[f'coded_{i + 1}' for i in range(CODING_DIM)])
    coded_path = FILE_PREFIX + 'coded_data.csv'
    coded_df.to_csv(coded_path, index=False)
    print(f"Закодированные данные сохранены в: {coded_path}")

    # 2. Декодированные данные
    decoded_data = decoder_model.predict(coded_data)
    decoded_df = pd.DataFrame(decoded_data, columns=X.columns)  #
    decoded_path = FILE_PREFIX + 'decoded_data.csv'
    decoded_df.to_csv(decoded_path, index=False)
    print(f"Декодированные данные сохранены в: {decoded_path}")

    # 3. Результаты регрессии
    regression_predictions = regression_model.predict(X)

    results_df = pd.DataFrame({
        'Actual': y_reg.values,
        'Predicted': regression_predictions.flatten()  #
    })
    results_path = FILE_PREFIX + 'regression_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Результаты регрессии сохранены в: {results_path}")

    print("\nВыполнение завершено")