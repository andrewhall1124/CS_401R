N = 7
r = 0.15
W = 30

E_Vk_xk = [0.0 for _ in range(N + 1)]

def value_function(E_Vk_xk, k, N, r, W):
    return ((E_Vk_xk[k + 1] ** 2) / (2 * W * (1 + r) ** (N - k))) + (W / 2) * (1 + r) ** (N- k)

E_Vk_xk[N] = W / 2

for k in range(N)[::-1]:
    E_Vk_xk[k] = value_function(E_Vk_xk, k, N, r, W)

def pretty_print(E_Vk_xk, fmt="{:>6.3f}"):
    header = "k    E_Vk_xk"

    print(header)
    for k, Vk_xk in enumerate(E_Vk_xk):
        print(f"{k}    {fmt.format(Vk_xk)}")

pretty_print(E_Vk_xk)