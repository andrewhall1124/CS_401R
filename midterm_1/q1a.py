import numpy as np

mat = np.array([
    [0, .7, .3, 0],
    [0, .2, 0, .8],
    [0, .9, 0, .1],
    [.5, 0, 0, .5]
])

# Example 1:
v_0 = np.array([.25, .25, .25, .25])

v_k = v_0
for _ in range(1000):
    v_k = v_k @ mat

print()
print("v_0 = [.25, .25, .25, .25]")
print("v_k = ", v_k)

# Example 2:
v_0 = np.array([1, 0, 0, 0])

v_k = v_0
for _ in range(1000):
    v_k = v_k @ mat

print()
print("v_0 = [1, 0, 0, 0]")
print("v_k = ", v_k)
