# ---------- Problem data ----------
C  = 3        # capacity (max inventory)
N  = 4        # horizon (stages k = 0..N)
cu = 3        # ordering cost per unit
cx = 1        # holding cost per unit
cd = 5        # lost-sales penalty per unit
cN = 4        # terminal (salvage) cost per leftover unit

demand_probs = {
    1: [0.20, 0.25, 0.25, 0.10],   # k=1
    2: [0.20, 0.25, 0.25, 0.40],   # k=2
    3: [0.40, 0.25, 0.25, 0.10],   # k=3
    4: [0.40, 0.25, 0.25, 0.40],   # k=4
}

# ---------- Helpers ----------
def next_state(x, u, w):
    """Inventory left after demand."""
    return max(0, x + u - w)

def stage_cost(k, x, u, w):
    """g_k(x,u,w) for k < N. At k==N we don't call this (terminal)."""
    shortage = max(0, w - (x + u))
    return cu * u + cx * x + cd * shortage

# ---------- DP tables ----------
# V[k][x] will hold the optimal value at stage k with inventory x
# pi[k][x] will hold the optimal action u at stage k with inventory x
V  = [[0.0 for _ in range(C + 1)] for _ in range(N + 1)]
pi = [[0   for _ in range(C + 1)] for _ in range(N + 1)]

# Terminal stage: V_N(x) = cN * x
for x in range(C + 1):
    V[N][x] = cN * x
    pi[N][x] = 0  # no action at terminal

# Backward induction for k = N-1 ... 0
for k in range(N - 1, -1, -1):
    # Use demand distribution for k+1 (since w_k is demand on day k+1)
    probs = demand_probs[k + 1]  # list of 4 probabilities for w=0..3
    for x in range(C + 1):
        best_u = 0
        best_val = float('inf')
        # Feasible orders: keep inventory within capacity
        for u in range(0, C - x + 1):
            # Expected total cost for this u
            expected = 0.0
            for w, p in enumerate(probs):
                if p == 0.0:
                    continue
                ns = next_state(x, u, w)
                g  = stage_cost(k, x, u, w) if k < N else 0.0
                expected += p * (g + V[k + 1][ns])
            if expected < best_val - 1e-12:
                best_val = expected
                best_u = u
        V[k][x]  = best_val
        pi[k][x] = best_u

# ---------- Pretty print ----------
def print_table(title, tbl, fmt="{:>6.3f}"):
    print(title)
    header = "k= " + " ".join(f"{k:>8}" for k in range(0, N + 1))
    print(header)
    for x in range(C + 1):
        row = [f"x={x:>1}"] + [fmt.format(tbl[k][x]) for k in range(0, N + 1)]
        print(" ".join(row))
    print()

print_table("Value function V_k(x):", V, fmt="{:>8.3f}")
print_table("Policy pi_k(x) [optimal u]:", pi, fmt="{:>8d}")