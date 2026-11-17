"""Table 3: decision histories, information sets and game states per configuration.

Implements Appendix C exactly. For an HxD N-player game there are B = H*D*N
bids, totally ordered, and h = C(H+D-1, D-1) isomorphic hands per player. With
G_z(b) = sum_{b' > b} g(b', 0, z) and G_0(B) = G_1(B) = 0,

    g(b, 0, 0) = (N-1) * (1 + G_0(b)) + 1 + G_1(b)        (1)
    g(b, 0, 1) = (N-1) * (1 + G_0(b))                     (2)
    P_B       = 1 + sum_{b=1..B} g(b, 0, 0)

iterated backward from b = B. Information sets are h * P_B (bid histories are
public and determine who acts, so an information set is the acting player's hand
paired with a decision history); game states are h^N * P_B.

Everything is exact Python integers -- P_B reaches ~10^306 at 8x10 5-player, so
floats would overflow. Run:  python paper/game_size.py
"""
import math

# (label, H hand_length, D num_digits, N players, published P_B from Table 3)
CONFIGS = [
    ("3x3 2-player", 3, 3, 2, "3.9e7"),
    ("3x3 3-player", 3, 3, 3, "3.0e15"),
    ("5x5 2-player", 5, 5, 2, "9.3e20"),
    ("8x10 3-player", 8, 10, 3, "2.0e137"),
    ("8x10 5-player", 8, 10, 5, "1.7e306"),
]


def decision_histories(H, D, N):
    """P_B by backward iteration of (1) and (2). Exact integers."""
    B = H * D * N
    G0 = 0          # sum of g(b',0,0) over b' > b, maintained as b decreases
    G1 = 0          # same for z = 1
    total = 0       # sum of g(b,0,0) over all b
    for _b in range(B, 0, -1):
        g0 = (N - 1) * (1 + G0) + 1 + G1
        g1 = (N - 1) * (1 + G0)
        total += g0
        G0 += g0    # these become G_z(b-1) on the next iteration
        G1 += g1
    return B, 1 + total


def sci(n, digits=2):
    """Scientific notation for an exact integer of arbitrary size."""
    if n == 0:
        return "0"
    e = len(str(n)) - 1
    mant = n / 10 ** e if e < 300 else int(str(n)[:digits + 2]) / 10 ** (digits + 1)
    return f"{mant:.{digits-1}f}e{e}"


def main():
    hdr = (f"{'Configuration':<15}{'B':>6}{'Hands h':>10}{'P_B':>12}"
           f"{'Info sets':>12}{'Game states':>13}   {'Table 3 P_B':>12}")
    print(hdr)
    print("-" * len(hdr))
    for label, H, D, N, published in CONFIGS:
        B, PB = decision_histories(H, D, N)
        h = math.comb(H + D - 1, D - 1)
        infosets = h * PB
        states = (h ** N) * PB
        print(f"{label:<15}{B:>6}{h:>10,}{sci(PB):>12}"
              f"{sci(infosets):>12}{sci(states):>13}   {published:>12}")
    print("\nP_B column should match the published one to the quoted precision.")


if __name__ == "__main__":
    main()
