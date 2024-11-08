from simulation import Simulation
from typing import Tuple
import numpy as np
import os, random
import multiprocessing as mp


def run_single_iter(
    nqubits: int,
    max_depth: int = 100,
    architecture: str = "linear",
    pbc=True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    if architecture not in ("linear", "brickwork", "random"):
        raise ValueError(
            "architecture should be either `linear`, `brickwork`, or `random`"
        )

    n = int(0.25 * (nqubits // 2) - 1)
    ms = list(range(n + 1, n + 27))
    ms.append(int(0.75 * (nqubits // 2) - 1))
    lams = [0.0, 0.05, 0.1, 0.5, 0.8, 1.0]

    string = np.zeros((len(lams), len(ms), max_depth + 1))
    trivial_string = np.zeros((len(lams), len(ms), max_depth + 1))
    zz = np.zeros((len(lams), len(ms), max_depth + 1))

    sims = [Simulation().initialize(nqubits) for _ in lams]
    for i in range(len(lams)):
        for j, m in enumerate(ms):
            s = sims[i]
            string[i, j, 0] += s.peek_string_order(n, m)
            trivial_string[i, j, 0] += s.peek_trivial_string_order(n, m)
            zz[i, j, 0] += s.peek_zz(n, m)

    for t in range(max_depth):
        if verbose and not t % 1000:
            print("starting depth", t)
        if architecture == "random":
            for i, lam in enumerate(lams):
                sims[i].E_lambda(random.randint(0, nqubits - 1), lam, pbc)
        else:
            if architecture == "linear":
                qubits = range(t % 2, nqubits, 2)
                # qubits = ((2 * t + (t // (nqubits // 2) % 2)) % nqubits,)
            elif architecture == "brickwork":
                qubits = range(t % 3, nqubits, 3)

            for j in qubits:
                for i, lam in enumerate(lams):
                    sims[i].E_lambda(j, lam, pbc)

        for i, lam in enumerate(lams):
            for j, m in enumerate(ms):
                s = sims[i]
                string[i, j, t + 1] += s.peek_string_order(n, m)
                trivial_string[i, j, t + 1] += s.peek_trivial_string_order(n, m)
                zz[i, j, t + 1] += s.peek_zz(n, m)

    return string, trivial_string, zz


def mp_run_single_iter(args) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return run_single_iter(*args)


def mixingtime_simulation(
    filename: str,
    nqubits: int = 100,
    iters: int = 100,
    pbc: bool = True,
):

    max_depth = int(nqubits**2 / 4)

    try:
        os.mkdir("analysis/mixingtime2.0/data")
    except FileExistsError:
        pass

    i = 0
    while f"{filename}.dat" in os.listdir("analysis/mixingtime2.0/data"):
        filename = filename + f"_{i}"
        i += 1
    with open(f"analysis/mixingtime2.0/data/{filename}.dat", "w+") as f:
        printf = lambda *args: print(*args, file=f)
        printf(f"nqubits = {nqubits}")
        printf(f"iters = {iters}")
        printf(f"pbc = {pbc}")
        printf(
            "\nData is the average value of the string order parameter as a function of (lam, m, depth) "
            "for lam in [0., .05, .1, .5, .8, 1.] and m in range(2, 27)\n"
            "random architecture"
        )
        printf("#" * 20)

        work_items = [(nqubits, max_depth, "random", pbc, True)] * iters
        with mp.Pool() as pool:
            output = list(pool.imap_unordered(mp_run_single_iter, work_items))
        # output = [
        #     run_single_iter(nqubits, max_depth, "random", pbc, True)
        #     for _ in range(iters)
        # ]

        string, trivial_string, zz = 0, 0, 0
        for s, ts, z in output:
            string += s
            trivial_string += ts
            zz += z
        string /= iters
        trivial_string /= iters
        zz /= iters

        printf(string.tolist())
        printf("&&&")
        printf(trivial_string.tolist())
        printf("***")
        printf(zz.tolist())


if __name__ == "__main__":
    import sys

    argv = sys.argv[1:]

    ext = argv[0] if len(argv) == 1 else ""
    # for nqubits in range(60, 81, 20):
    #     mixingtime_simulation(
    #         f"nqubits_{nqubits}" + ext, nqubits=nqubits, iters=10000, pbc=False
    #     )
    for nqubits in range(100, 121, 20):
        mixingtime_simulation(
            f"nqubits_{nqubits}" + ext, nqubits=nqubits, iters=10000, pbc=False
        )
