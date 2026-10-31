import numpy as np
import pandas as pd
from scipy import linalg, stats

def print_matrix(name, M):
    df = pd.DataFrame(np.round(M, 4))
    print(f"\n{name}:")
    print(df.to_string(index=False, header=False))

#Задание 1
A = np.array([
    [ 4.0, -3.4,  1.0,  1.8],
    [-2.0,  8.0,  0.0, -9.0],
    [ 2.0, -2.5, -7.0,  4.0],
    [ 1.0, -1.0,  5.0,  1.0]
], dtype=float)
print_matrix("A", A)

#Задание 2
P, L, U = linalg.lu(A)
print_matrix("P", P)
print_matrix("L", L)
print_matrix("U", U)

#Задание 3
det_A = np.linalg.det(P) * np.prod(np.diag(L)) * np.prod(np.diag(U))
print(f"\nОпределитель A = {det_A:.6f}")

#Задание 4
np.random.seed(42)
n = 100
low, high = 0, 100

uniform_sample = np.random.randint(low, high + 1, n)

mu = (low + high) / 2
sigma = (high - low) / 6
normal_sample = np.clip(np.round(np.random.normal(mu, sigma, n)),
                        low, high).astype(int)


#Задание 5
def describe(sample, name):
    return {
        'Выборка':               name,
        'Среднее':               round(float(np.mean(sample)), 4),
        'Мода':                  int(stats.mode(sample, keepdims=False).mode),
        'Медиана':               round(float(np.median(sample)), 4),
        'Минимум':               int(np.min(sample)),
        'Максимум':              int(np.max(sample)),
        'Стд. отклонение':       round(float(np.std(sample)), 4),
    }


df_stats = pd.DataFrame([
    describe(uniform_sample, 'Равномерное'),
    describe(normal_sample,  'Нормальное'),
])
print("\nСтатистики выборок:")
print(df_stats.to_string(index=False))

#Задание 6
def chi2_pvalue(sample, low, high, bins=10):
    edges = np.linspace(low, high + 1, bins + 1)
    observed, _ = np.histogram(sample, bins=edges)
    expected = np.full(bins, observed.sum() / bins, dtype=float)
    _, p = stats.chisquare(observed, expected)
    return float(p)

print(f"\np-value (равномерная выборка): {chi2_pvalue(uniform_sample, low, high):.6f}")
print(f"p-value (нормальная выборка):  {chi2_pvalue(normal_sample,  low, high):.6f}")
