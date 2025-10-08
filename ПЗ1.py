import math
import random
from collections import defaultdict
from typing import List, Tuple, Union


# Задача 1
def count_vowels(s: str) -> int:
    vowels = "aeiou"
    return sum(1 for char in s.lower() if char in vowels)

print(f"Задача 1: {count_vowels('Hello World')}")
print(f"Задача 1: {count_vowels('Rhythm')}")
print(f"Задача 1: {count_vowels('AEIOU aeiou')}")



# Задача 2
def are_chars_unique(s: str) -> bool:
    return len(s) == len(set(s))

print(f"Задача 2: {are_chars_unique('abcdefg')}")
print(f"Задача 2: {are_chars_unique('hello')}")
print(f"Задача 2: {are_chars_unique('a b c')}")



# Задача 3
def count_set_bits(n: int) -> int:
    return bin(n).count('1')

print(f"Задача 3: {count_set_bits(13)}")
print(f"Задача 3: {count_set_bits(16)}")
print(f"Задача 3: {count_set_bits(15)}")



# Задача 4
def multiplicative_persistence(n: int) -> int:
    steps = 0
    while n >= 10:
        steps += 1
        product = 1
        for digit in str(n):
            product *= int(digit)
        n = product
    return steps

print(f"Задача 4: {multiplicative_persistence(39)}")
print(f"Задача 4: {multiplicative_persistence(999)}")
print(f"Задача 4: {multiplicative_persistence(679)}")



# Задача 5
def mean_squared_error(vec1: List[int], vec2: List[int]) -> float:
    squared_diff_sum = sum((v1 - v2) ** 2 for v1, v2 in zip(vec1, vec2))
    return squared_diff_sum / len(vec1)

print(f"Задача 5: {mean_squared_error([1, 2, 3], [1, 2, 3])}")
print(f"Задача 5: {mean_squared_error([1, 2, 3], [3, 2, 1])}")
print(f"Задача 5: {mean_squared_error([-1, 0, 1], [1, 0, -1]):.2f}")



# Задача 6
def calculate_stats(numbers: List[float]) -> Tuple[float, float]:
    n = len(numbers)
    if n == 0:
        return (0.0, 0.0)
    mean = sum(numbers) / n
    variance = sum([(x - mean) ** 2 for x in numbers]) / n
    std_dev = variance ** 0.5
    return (mean, std_dev)

mean1, std1 = calculate_stats([1, 2, 3, 4, 5, 6])
print(f"Задача 6: Мат. ожидание = {mean1:.2f}, СКО = {std1:.2f}")

mean2, std2 = calculate_stats([5, 5, 5, 5, 5])
print(f"Задача 6: Мат. ожидание = {mean2:.2f}, СКО = {std2:.2f}")



# Задача 7
def prime_factorization_format(n: int) -> str:
    factors = defaultdict(int)
    d = 2
    temp_n = n
    while d * d <= temp_n:
        while temp_n % d == 0:
            factors[d] += 1
            temp_n //= d
        d += 1
    if temp_n > 1:
        factors[temp_n] += 1
    result = []
    for factor, power in sorted(factors.items()):
        if power == 1:
            result.append(f"({factor})")
        else:
            result.append(f"({factor}**{power})")
    return "".join(result)

print(f"Задача 7: {prime_factorization_format(86240)}")
print(f"Задача 7: {prime_factorization_format(90)}")
print(f"Задача 7: {prime_factorization_format(13)}")



# Задача 8
def get_network_info(ip_str: str, mask_str: str) -> Tuple[str, str]:
    def ip_to_int(ip_s):
        octets = ip_s.split('.')
        return int(octets[0]) << 24 | int(octets[1]) << 16 | int(octets[2]) << 8 | int(octets[3])
    def int_to_ip(ip_int):
        return f"{(ip_int >> 24) & 255}.{(ip_int >> 16) & 255}.{(ip_int >> 8) & 255}.{ip_int & 255}"
    ip_int = ip_to_int(ip_str)
    mask_int = ip_to_int(mask_str)
    network_int = ip_int & mask_int
    broadcast_int = network_int | (~mask_int & 0xffffffff)
    return int_to_ip(network_int), int_to_ip(broadcast_int)

net, broad = get_network_info("192.168.1.130", "252.255.255.192")
print(f"Задача 8: Сеть: {net}, Широковещательный: {broad}")

net, broad = get_network_info("10.150.20.3", "255.0.0.0")
print(f"Задача 8: Сеть: {net}, Широковещательный: {broad}")



# Задача 9
def can_build_pyramid(n: int) -> Union[str, int]:
    current_sum = 0
    k = 0
    while current_sum < n:
        k += 1
        current_sum += k*k
    return k if current_sum == n else "It is impossible"

print(f"Задача 9: {can_build_pyramid(30)}")
print(f"Задача 9: {can_build_pyramid(14)}")
print(f"Задача 9: {can_build_pyramid(1)}")



# Задача 10
def is_balanced(n: int) -> bool:
    s = str(n)
    length = len(s)
    mid = length // 2
    if length < 3:
        return True
    if length % 2 == 1:
        left_part = s[:mid]
        right_part = s[mid+1:]
    else:
        left_part = s[:mid-1]
        right_part = s[mid+1:]
    left_sum = sum(int(digit) for digit in left_part)
    right_sum = sum(int(digit) for digit in right_part)
    return left_sum == right_sum

print(f"Задача 10: {is_balanced(23441)}")
print(f"Задача 10: {is_balanced(123456)}")
print(f"Задача 10: {is_balanced(13722731)}")



# Задача 11
def stratified_split(M: List[List], r: float) -> Tuple[List[List], List[List]]:
    groups = defaultdict(list)
    for row in M:
        groups[row[0]].append(row)
    split1, split2 = [], []
    for _, rows in groups.items():
        random.shuffle(rows)
        split_point = round(len(rows) * r)
        split1.extend(rows[:split_point])
        split2.extend(rows[split_point:])
    random.shuffle(split1)
    random.shuffle(split2)
    return split1, split2
data_matrix = [['a', 1], ['a', 2], ['a', 3], ['a', 4], ['b', 5], ['b', 6], ['b', 7], ['b', 8]]

s1, s2 = stratified_split(data_matrix, 0.5)
print(f"Задача 11: Выборка 1 ({len(s1)} эл.): {sorted(s1)}")
print(f"Задача 11: Выборка 2 ({len(s2)} эл.): {sorted(s2)}")

s3, s4 = stratified_split(data_matrix, 0.75)
print(f"Задача 11: Выборка 1 ({len(s3)} эл.): {sorted(s3)}")
print(f"Задача 11: Выборка 2 ({len(s4)} эл.): {sorted(s4)}")