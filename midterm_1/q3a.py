import numpy as np
from rich import print

# Parameters
N = 7

# Q(state weight matrix) and R(control weight matrix)
Q = np.array([[5, -4, 0], [-4, 16, 8], [0, 8, 5]])
R = np.array([[5, 0], [0, 2]])

# State transition matrices
A = np.array([[1, 2, -1], [-4, 1, 2], [-1, -1, 3]])
B = np.array([[1, 2], [5, -1], [-1, 3]])

# X initial state
x0 = np.array([20, 20, 20])

P = [None] * (N + 1)
P[N] = Q
K = [None] * N

# Compute P(cost to go matrix) and K(feedback gain matrix)
for k in range(N - 1, -1, -1):
    K[k] = np.linalg.inv(R + B.T @ P[k + 1] @ B) @ B.T @ P[k + 1] @ A
    P[k] = Q + K[k].T @ R @ K[k] + (A - B @ K[k]).T @ P[k + 1] @ (A - B @ K[k])

x = np.zeros((3, N + 1))
u = np.zeros((2, N))
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
print("\nCost to go matrices")
for k in range(N + 1):
    print(f"P_{k} =\n {P[k].round(2)}")

print("\nValue to go")
for k in range(N + 1):
    print(f"V_{k} = {V[k]:.2f}")

print("\nStage cost")
for k in range(N + 1):
    print(f"g_{k} = {g[k]:.2f}")

print(f"\nTotal Cost = {J:.2f}")

print("\nOptimal Actions")
for k in range(N):
    print(f"u_{k} = {u[:, k].round(3)}")

print("\nFinal State")
print(f"x_7 = {x[:,7].round(3)}")

