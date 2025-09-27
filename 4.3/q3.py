import numpy as np
from rich import print
import matplotlib.pyplot as plt

# Parameters
N = 4

# Q(state weight matrix) and R(control weight matrix)
Q = np.array([[5, -4, 0], [-4, 16, 0], [0, 0, 0]])
R = np.array([[1, 0], [0, 3]])

# State transition matrices
A = np.array([[1, 2, 0], [-4, 1, 0], [0, -1, 3]])
B = np.array([[1, 2], [0, -1], [-1, 0]])

# X initial state
x0 = np.array([-13, 25, 20])

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
for k in range(N + 1):
    print(f"P_{k} =\n {P[k].round(2)}")

for k in range(N + 1):
    print(f"V_{k} = {V[k]:.2f}")

for k in range(N + 1):
    print(f"g_{k} = {g[k]:.2f}")

for k in range(N + 1):
    print(f"x_{k} = {x[:, k].round(2)}")

print(f"Total Cost = {J:.2f}")

# Plot the path of x
time_steps = np.arange(N + 1)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)

plt.plot(time_steps, x[0, :], 'red')
plt.title('State x_1 vs Time')
plt.xlabel('Time Step k')
plt.ylabel('x_1')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(time_steps, x[1, :], 'green')
plt.title('State x_2 vs Time')
plt.xlabel('Time Step k')
plt.ylabel('x_2')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(time_steps, x[2, :], 'blue')
plt.title('State x_3 vs Time')
plt.xlabel('Time Step k')
plt.ylabel('x_3')
plt.grid(True)

plt.tight_layout()
plt.show()

