import numpy as np
from simulation import run_simulation


def run_and_write_avg_simulation(
    filename: str,
    nqubits: int,
    max_depth: int = 100,
    iters: int = 10,
    architecture: str = "linear",
    lam: float = 0,
    pbc: bool = True,
) -> None:
    p1s = np.linspace(0.0, 0.05, 20)
    p2s = np.linspace(0.0, 0.05, 20)

    with open(f"analysis/mipt/data/{filename}.dat", "w+") as f:
        printf = lambda *args: print(*args, file=f)
        printf(f"nqubits = {nqubits}")
        printf(f"max_depth = {max_depth}")
        printf(f"iters = {iters}")
        printf(f"architecture = {architecture}")
        printf(f"lam = {lam}")
        printf(f"pbc = {pbc}")
        printf("p1, p2, string, trivial_string")
        printf("#" * 20)

        for p1 in p1s:
            for p2 in p2s:
                string, trivial_string = run_simulation(
                    nqubits, p1, p2, max_depth, iters, architecture, lam, pbc
                )
                string = np.mean(string[max_depth // 10 :])
                trivial_string = np.mean(trivial_string[max_depth // 10 :])
                printf(f"{p1}, {p2}, {string}, {trivial_string}")


if __name__ == "__main__":
    # for nqubits in range(50, 301, 50):
    #     print("Starting nqubits =", nqubits)
    #     run_and_write_avg_simulation(f"nqubits_{nqubits}", nqubits)
    run_and_write_avg_simulation(f"TEST", 10)
