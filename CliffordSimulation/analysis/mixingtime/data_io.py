from simulation import run_simulation
import os


def mixingtime_simulation(
    filename: str,
    num_qubits: list,
    iters: int = 10,
    lam: float = 1,
    pbc: bool = False,
):

    # max_depth = max(num_qubits) ** 2
    max_depth = int(max(num_qubits) ** 2 / 4)

    try:
        os.mkdir("analysis/mixingtime/data")
    except FileExistsError:
        pass

    i = 0
    while f"{filename}.dat" in os.listdir("analysis/mixingtime/data"):
        filename = filename + f"_{i}"
        i += 1
    with open(f"analysis/mixingtime/data/{filename}.dat", "w+") as f:
        printf = lambda *args: print(*args, file=f)
        printf(f"iters = {iters}")
        printf(f"lam = {lam}")
        printf(f"pbc = {pbc}")
        printf(
            "\nData is the average value of the string order parameter as a function of depth"
        )
        printf("#" * 20)
        for nqubits in num_qubits:
            # print("\tStarting", nqubits)
            string, trivial_string = run_simulation(
                nqubits, 0, 0, max_depth, iters, "random", lam, pbc
            )
            printf(f"string order: qubits {nqubits}")
            printf(string.tolist())
            printf(f"trivial string order: qubits {nqubits}")
            printf(trivial_string.tolist())


if __name__ == "__main__":
    import sys

    argv = sys.argv[1:]

    ext = argv[0] if len(argv) == 1 else ""
    lams = [0.0, 0.05, 0.1, 0.5, 0.8, 1.0]
    for lam in lams:
        # print("Starting lam =", lam)
        mixingtime_simulation(
            f"lam_eq_{int(100*lam)}" + ext,
            list(range(60, 121, 20)),
            iters=1000,
            lam=lam,
        )
