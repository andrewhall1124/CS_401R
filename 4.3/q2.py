import numpy as np
from rich import print
# Parameters
N = 9
l = 1.0
g = 9.8
m = 0.3

# Q(state weight matrix) and R(control weight matrix)
Q = np.array([[1, 0], [0, 0]])
R = np.array([[1, 0], [0, 2]])

# State transition matrices
A = np.array([[1, 2], [-4, 1]])
B = np.array([[1, 2], [3, 2]])

# X initial state
x0 = np.array([5, 10])

P = [None] * (N + 1)
P[N] = Q
K = [None] * N

def V(x, u):
    return x.T @ Q @ x + u.T @ R @ u

# Compute P(cost to go matrix) and K(feedback gain matrix)
for k in range(N - 1, -1, -1):
    K[k] = np.linalg.inv(R + B.T @ P[k + 1] @ B) @ B.T @ P[k + 1] @ A
    P[k] = Q + K[k].T @ R @ K[k] + (A - B @ K[k]).T @ P[k + 1] @ (A - B @ K[k])

x = np.zeros((2, N + 1))
u = np.zeros((2, N))  # u should be 2D since B is 2x2
x[:, 0] = x0

# Compute u(control inputs) and x(state)
for k in range(N):
    u[:, k] = -K[k] @ x[:, k]
    x[:, k+1] = A @ x[:, k] + B @ u[:, k]

g = np.zeros(N + 1)
V = np.zeros(N + 1)

# Stage costs g_k (cost incurred at step k)
for k in range(N):
    g[k] = x[:,k].T @ Q @ x[:,k] + u[:,k].T @ R @ u[:,k]

g[N] = x[:,N].T @ Q @ x[:,N]

# Value-to-go V_k (total cost from step k to end)
V[N] = x[:,N].T @ Q @ x[:,N]  # Terminal cost
for k in range(N-1, -1, -1):
    V[k] = g[k] + V[k+1]


# Total cost
J = V[0]

# Ouput
for k in range(N + 1):
    print(f"P_{k} =\n {P[k].round(2)}")

for k in range(N + 1):
    print(f"V_{k} = {V[k]:.2f}")

for k in range(N + 1):
    print(f"g_{k} = {g[k]:.2f}")

print(f"Total Cost = {J:.2f}")

