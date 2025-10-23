import random
import matplotlib.pyplot as plt
from typing import Callable

random.seed(42)

N = 100

def x_1(n: int | None = None) -> float:
    return [5, 2, 9, 10, 1, 3][n]

def x_2(n: int | None = None) -> float:
    return random.uniform(50, 100)

def nu_1(n: float) -> float:
    return 1 / n

def nu_2(n: float) -> float:
    return (0.5) / (n + 1)

def nu_3(n: float) -> float:
    return 1 / ((n + 3) ** 2)


def run(x: Callable, nu: Callable, N: int) -> list[float]:
    mu = [None] * N
    for n in range(N):
        if n == 0:
            mu[n] = x(n)
        else:
            mu[n] = mu[n-1] + nu(n+1) * (x(n) - mu[n-1])
    
    return mu

N = 100
x = list(range(N))
mu_1 = run(x_2, nu_1, N)
mu_2 = run(x_2, nu_2, N)
mu_3 = run(x_2, nu_3, N)

plt.plot(x, mu_1, label="nu_1")
plt.plot(x, mu_2, label="nu_2")
plt.plot(x, mu_3, label="nu_3")

plt.title("Comparison of nu")

plt.xlabel("n")
plt.ylabel("mu_n")

plt.axhline(y=75, color='red')

plt.legend()

plt.show()