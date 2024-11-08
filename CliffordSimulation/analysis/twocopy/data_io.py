# import matplotlib.pyplot as plt
# import numpy as np
from run_simulation import run_two_copy_simulation
import os


# def heatmap(x, y, z, xlabel="", ylabel="", title=""):
#     # generate 2 2d grids for the x & y bounds
#     y, x = np.meshgrid(y, x)

#     fig, ax = plt.subplots()

#     c = ax.pcolormesh(x, y, z, cmap="RdBu")
#     ax.set_title(title)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.axis([x.min(), x.max(), y.min(), y.max()])
#     fig.colorbar(c, ax=ax)

#     plt.show()


def run_and_write_avg_simulation(
    filename: str,
    nqubits: int,
    max_depth: int = 40,
    iters: int = 10,
    architecture: str = "random",
    lam: float = 1.0,
    pbc: bool = True,
) -> None:
    # p1s = np.linspace(0.0, 0.02, 3)
    # p2s = np.linspace(0.0, 0.02, 3)
    p1s = [0.0]
    p2s = [0.0]

    try:
        os.mkdir("analysis/twocopy/data")
    except FileExistsError:
        pass

    i = 0
    while f"{filename}.dat" in os.listdir("analysis/twocopy/data"):
        filename = filename + f"_{i}"
        i += 1

    with open(f"analysis/twocopy/data/{filename}.dat", "w+") as f:
        printf = lambda *args: print(*args, file=f)
        printf(f"nqubits = {nqubits}")
        printf(f"max_depth = {max_depth}")
        printf(f"iters = {iters}")
        printf(f"architecture = {architecture}")
        # printf("p1, p2, depth, purity, S, W")
        printf(
            "purity, then endpoints, then string S, then string W, then trivial S, then trivial W, then ZZZZ"
        )
        printf("#" * 20)

        # purity, S, W = run_two_copy_simulation(
        #     nqubits, p1, p2, max_depth, iters, architecture
        # )
        (
            purity,
            endpoints,
            string_S,
            string_W,
            trivial_string_S,
            trivial_string_W,
            ZZZZ,
        ) = run_two_copy_simulation(
            nqubits, 0, 0, max_depth, iters, architecture, lam, pbc
        )
        printf(purity.tolist())
        printf(endpoints.tolist())
        printf(string_S.tolist())
        printf(string_W.tolist())
        printf(trivial_string_S.tolist())
        printf(trivial_string_W.tolist())
        printf(ZZZZ.tolist())
        # for t in range(len(purity)):
        # printf(f"{p1}, {p2}, {t}, {purity[t]}, {S[t]}, {W[t]}")
        # printf(f"{p1}, {p2}, {t}, {purity[t]}, {endpoints[t]}")


if __name__ == "__main__":

    for nqubits in range(6, 21, 2):
        # for nqubits in range(18, 21, 2):
        for lam in (0, 0.05, 1):
            # for nqubits, lam in [(18, 1), (20, 0), (20, 0.05), (20, 1)]:
            print("Starting nqubits =", nqubits, "lam =", lam)
            run_and_write_avg_simulation(
                f"nqubits_{nqubits}_lam_{int(100*lam)}",
                nqubits,
                # iters=(1 << nqubits) * 20,
                iters=10000,
                max_depth=2 * nqubits**2,
                lam=lam,
            )
