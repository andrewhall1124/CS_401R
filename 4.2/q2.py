import random

def secretary_policy(candidates, r=37):
    """Apply the look-then-leap policy. 
    candidates is a list of unique integers (1=best)."""
    best_so_far = min(candidates[:r])  # best among first r
    for i in range(r, len(candidates)):
        if candidates[i] < best_so_far:  # new best found
            return candidates[i]
    return candidates[-1]  # if none chosen, hire last

# Simulation
N = 100          # number of candidates
trials = 1000    # number of random sequences
r = 37           # look-then-leap threshold

success = 0
for _ in range(trials):
    seq = list(range(1, N+1))
    random.shuffle(seq)
    chosen = secretary_policy(seq, r)
    if chosen == 1:   # did we pick the best?
        success += 1

print(f"Out of {trials} trials, selected the best {success} times.")
print(f"Success rate: {success/trials:.3f}")