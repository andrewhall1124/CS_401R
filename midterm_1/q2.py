N = 10
r = 0.1
E_W = 100 / 3

E_Vk_xk = [0.0] * (N + 1)

def value_function(E_Vk_xk, k, N, r):
    return (
        (
            (E_Vk_xk[k + 1] ** 3) / 
            (7500 * (1 + r) ** (2 * (N - k)))
        ) + 
        (100 / 3) * (1 + r) ** (N- k)
)

E_Vk_xk[N] = 100 / 3

for k in range(N)[::-1]:
    E_Vk_xk[k] = value_function(E_Vk_xk, k, N, r)

def pretty_print(E_Vk_xk):
    header = "k    E_Vk_xk"

    print(header)
    for k, Vk_xk in enumerate(E_Vk_xk):
        print(f"{k}    {round(Vk_xk, 2)}")

pretty_print(E_Vk_xk)

E_Vkp1_xkp1 = [0.0] * (N + 1)
W = [10.33, 7.25, 17.15, 5.12, 33.21, 8.56, 43.32, 31.44, 22.90, 50, 0, 0]
X_fv = [0.0] * (N + 1)
X = [0.0] * (N + 1)
policy = [''] * (N + 1)

for k in range(1, N + 1):
    X[k] = W[k-1]

for k in range(N):
    E_Vkp1_xkp1[k] = E_Vk_xk[k + 1]
    X_fv[k] = X[k] * (1 + r) ** (N - k)
    
    if 'sell' not in policy:
        if E_Vkp1_xkp1[k] <= X_fv[k]:
                policy[k] = 'sell'
        else:
            policy[k] = 'hold'

def pretty_print_table(E_Vkp1_xkp1, X, X_fv, policy, fmt="{:.2f}"):
    """Pretty print E[V_{k+1}] and w_k values"""
    header = f"\n{'k':<4} {'E[V_k+1(x_k+1)]':<16} {'x_k':<6} {'x_fv':<6} {'u_k':<4}"
    print(header)    
    for k in range(len(E_Vkp1_xkp1)):
        a = fmt.format(E_Vkp1_xkp1[k])
        b = fmt.format(X[k])
        c = fmt.format(X_fv[k])
        d = policy[k]
        print(f"{k:<4} {a:<16} {b:<6} {c:<6} {d:<4}")


pretty_print_table(E_Vkp1_xkp1, X, X_fv, policy)
