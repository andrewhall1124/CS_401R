import numpy as np

# Transition matrix P
P = np.array([
    [0, .9, .1, 0],
    [.9, 0, 0, .1],
    [0, .1, .9, 0],
    [0, 0, .9, .1]
])

n = P.shape[0]

print("Closed Form Method:")
# pi @ P = pi
# (P - I)'pi = 0
# Add constraint: sum(pi) = 1
A = (P - np.eye(n)).T
A[-1] = np.ones(n)  # Replace last equation with sum constraint

b = np.zeros(n)
b[-1] = 1  # sum(pi) = 1
pi = np.linalg.solve(A, b)

print(pi.round(2))

print("Iterative Method:")
v = np.array([0, 0, 0, 1])

for _ in range(200):
    v = v @ P
v = v / sum(v)

print(v.round(2))