import numpy as np
import matplotlib.pyplot as plt

# Задача 1
def sum_of_products(matrices, vectors):
    products = matrices @ vectors
    result_vector = products.sum(axis=0)
    return result_vector

p, n = 3, 4
input_matrices = np.arange(p * n * n).reshape(p, n, n)
input_vectors = np.arange(p * n).reshape(p, n, 1)
np.save('task_1_matrices.npy', input_matrices)
np.save('task_1_vectors.npy', input_vectors)
loaded_matrices = np.load('task_1_matrices.npy')
loaded_vectors = np.load('task_1_vectors.npy')
result_1 = sum_of_products(loaded_matrices, loaded_vectors)
np.save('task_1_result.npy', result_1)

print("Задача 1 ")
print("Результат сохранен в 'task_1_result.npy'")
print("Результат:\n", result_1)


# Задача 2
def vector_to_binary_matrix(vector):
    if vector.size == 0:
        return np.array([[]])
    max_val = np.max(vector)
    width = int(np.ceil(np.log2(max_val + 1))) if max_val > 0 else 1
    binary_strings = [list(np.binary_repr(x, width=width)) for x in vector]
    return np.array(binary_strings, dtype=int)

input_vector_2 = np.array([5, 2, 7, 1, 8])
np.save('task_2_vector.npy', input_vector_2)
loaded_vector_2 = np.load('task_2_vector.npy')
result_2 = vector_to_binary_matrix(loaded_vector_2)
np.save('task_2_result.npy', result_2)

print("\n Задача 2 ")
print(f"Вектор: {loaded_vector_2}")
print("Бинарная матрица:\n", result_2)


# Задача 3
def get_unique_rows(matrix):
    return np.unique(matrix, axis=0)

input_matrix_3 = np.array([[0, 1, 2], [3, 4, 5], [0, 1, 2], [6, 7, 8], [3, 4, 5]])
np.save('task_3_matrix.npy', input_matrix_3)
loaded_matrix_3 = np.load('task_3_matrix.npy')
result_3 = get_unique_rows(loaded_matrix_3)
np.save('task_3_result.npy', result_3)

print("\n Задача 3 ")
print("Исходная матрица:\n", loaded_matrix_3)
print("Уникальные строки:\n", result_3)


# Задача 4
def analyze_normal_matrix(M, N, show_plots=True):
    matrix = np.random.randn(M, N)
    col_means = np.mean(matrix, axis=0)
    col_vars = np.var(matrix, axis=0)
    if show_plots:
        print("\n Задача 4 (графики)")
        for i, row in enumerate(matrix):
            plt.figure()
            plt.hist(row, bins='auto')
            plt.title(f'Гистограмма для строки {i+1}')
            plt.show()
    return matrix, col_means, col_vars

matrix_4, means_4, variances_4 = analyze_normal_matrix(M=4, N=5, show_plots=False)
np.save('task_4_matrix.npy', matrix_4)
np.savetxt('task_4_means.txt', means_4)
np.savetxt('task_4_variances.txt', variances_4)

print("\n Задача 4")
print("Матрица сохранена в 'task_4_matrix.npy'")
print("Мат. ожидания столбцов:\n", means_4)
print("Дисперсии столбцов:\n", variances_4)


# Задача 5
def chessboard_matrix(M, N, a, b):
    indices_sum = np.indices((M, N)).sum(axis=0)
    return np.where(indices_sum % 2 == 0, a, b)

result_5 = chessboard_matrix(M=8, N=8, a=1, b=0)
np.savetxt('task_5_result.txt', result_5, fmt='%d')

print("\n Задача 5")
print("Шахматная матрица сохранена в 'task_5_result.txt'")
print("Матрица 8x8:\n", result_5)


# Задача 6
def create_circle_image(height, width, radius, color):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    center_y, center_x = height // 2, width // 2
    y, x = np.ogrid[:height, :width]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    image[mask] = color
    return image

result_image_6 = create_circle_image(height=200, width=200, radius=80, color=[255, 0, 0])
np.save('task_6_result.npy', result_image_6)

print("\n Задача 6 ")
print("Создан тензор изображения. Результат в 'task_6_result.npy'")


# Задача 7
def standardize_tensor(tensor):
    mean = np.mean(tensor)
    std = np.std(tensor)
    epsilon = 1e-7
    return (tensor - mean) / (std + epsilon)

input_tensor_7 = np.array([1, 2, 5, 10, 20], dtype=float)
np.save('task_7_tensor.npy', input_tensor_7)
loaded_tensor_7 = np.load('task_7_tensor.npy')
result_7 = standardize_tensor(loaded_tensor_7)
np.save('task_7_result.npy', result_7)

print("\n Задача 7 ")
print("Исходный тензор:", loaded_tensor_7)
print("Стандартизированный тензор:\n", result_7)


# Задача 8
def extract_patch(matrix, center_coords, patch_size, fill_value=0):
    patch_h, patch_w = patch_size
    pad_h, pad_w = patch_h // 2, patch_w // 2
    padded_matrix = np.pad(matrix, ((pad_h, pad_h), (pad_w, pad_w)),
                           mode='constant', constant_values=fill_value)
    center_y, center_x = center_coords
    new_y, new_x = center_y + pad_h, center_x + pad_w
    return padded_matrix[new_y - pad_h : new_y + pad_h + 1,
                         new_x - pad_w : new_x + pad_w + 1]

input_matrix_8 = np.arange(25).reshape(5, 5)
np.save('task_8_matrix.npy', input_matrix_8)
loaded_matrix_8 = np.load('task_8_matrix.npy')
result_8 = extract_patch(loaded_matrix_8, center_coords=(0, 0), patch_size=(3, 3))
np.save('task_8_result.npy', result_8)

print("\n Задача 8 ")
print("Исходная матрица:\n", loaded_matrix_8)
print("Патч (3, 3) с центром в (0, 0):\n", result_8)


# Задача 9
def row_wise_mode(matrix):
    modes = [vals[np.argmax(counts)] for row in matrix for vals, counts in [np.unique(row, return_counts=True)]]
    return np.array(modes)

input_matrix_9 = np.array([[1, 2, 1, 3, 1], [5, 5, 4, 4, 5], [0, 6, 7, 7, 7]])
np.save('task_9_matrix.npy', input_matrix_9)
loaded_matrix_9 = np.load('task_9_matrix.npy')
result_9 = row_wise_mode(loaded_matrix_9)
np.save('task_9_result.npy', result_9)

print("\n Задача 9 ")
print("Исходная матрица:\n", loaded_matrix_9)
print("Самые частые значения в строках:\n", result_9)


# Задача 10
def weighted_channel_sum(image, weights):
    return np.dot(image, weights)

image_tensor_10 = np.random.randint(0, 256, (5, 5, 3))
weights_vector_10 = np.array([0.2989, 0.5870, 0.1140])
np.save('task_10_image.npy', image_tensor_10)
np.save('task_10_weights.npy', weights_vector_10)
loaded_image_10 = np.load('task_10_image.npy')
loaded_weights_10 = np.load('task_10_weights.npy')
result_10 = weighted_channel_sum(loaded_image_10, loaded_weights_10)
np.save('task_10_result.npy', result_10)

print("\n Задача 10 ")
print("Форма исходного тензора:", loaded_image_10.shape)
print("Форма результирующей матрицы:", result_10.shape)
print("Результат сохранен в 'task_10_result.npy'")