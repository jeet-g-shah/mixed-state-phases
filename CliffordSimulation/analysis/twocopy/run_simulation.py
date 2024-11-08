import stim
import numpy as np
import random
from typing import Tuple
from simulation import Simulation
import multiprocessing as mp


# def two_copy_obs_squared(
#     u: Simulation,
#     v: Simulation,
#     pauli: stim.PauliString,
#     *,
#     endian: str = "little",
# ) -> float:
#     if not pauli:
#         return (u.state_vector(endian=endian).conj().T @ v.state_vector(endian=endian))[
#             0, 0
#         ]
#     return abs((
#         u.state_vector(endian=endian).conj().T
#         @ pauli.to_unitary_matrix(endian=endian)
#         @ v.state_vector(endian=endian)
#     )[0, 0]) ** 2


def two_copy_obs_squared(
    u: Simulation,
    v: Simulation,
    pauli: stim.PauliString = None,
) -> float:
    # compute |<u|O|v>|^2
    phi = v.copy()
    psi = u  # .copy()
    if pauli is not None:
        phi.do_pauli_string(pauli)
    # now just compute |<psi|phi>|^2
    H = psi.canonical_stabilizers()
    pval = 1.0
    for h in H:
        m = phi.peek_observable_expectation(h)
        if m == -1:
            return 0.0
        elif m == 0:
            pval /= 2.0
        phi.postselect_observable(h, desired_value=False)
    return pval


def two_copy_S(u: Simulation, v: Simulation, n: int, m: int, trivial: bool = False):
    # deal with zero vs one indexing
    if not u.nqubits == v.nqubits:
        raise ValueError("Different number of qubits")
    nqubits = u.nqubits
    n, m = min(n, m), max(n, m)
    if n <= 0 or 2 * m + 2 >= nqubits:
        raise ValueError("n, m = %d, %d are not in the required range" % (n, m))

    paulis = [0] * (2 * n - 2) + [0 if trivial else 3]
    for j in range(n, m):
        paulis.append(1)
        paulis.append(0)
    paulis.append(1)
    paulis.append(0 if trivial else 3)
    paulis += [0] * (nqubits - len(paulis))
    S = stim.PauliString(paulis)

    # <u|S|v><v|u>

    H = v.canonical_stabilizers()
    pval = 1.0
    for h in H:
        m = u.peek_observable_expectation(h)
        if m == -1:
            return 0.0
        elif m == 0:
            pval /= 2.0
        u.postselect_observable(h, desired_value=False)
    m = u.peek_observable_expectation(S)
    return m * pval


def two_copy_W(u: Simulation, v: Simulation, n: int, m: int, trivial: bool = False):
    # deal with zero vs one indexing
    if not u.nqubits == v.nqubits:
        raise ValueError("Different number of qubits")
    nqubits = u.nqubits
    n, m = min(n, m), max(n, m)
    if n <= 0 or 2 * m + 2 >= nqubits:
        raise ValueError("n, m = %d, %d are not in the required range" % (n, m))

    paulis = [0] * (2 * n - 1) + [0 if trivial else 3]
    for j in range(n, m):
        paulis.append(1)
        paulis.append(0)
    paulis.append(1)
    paulis.append(0 if trivial else 3)
    paulis += [0] * (nqubits - len(paulis))

    return two_copy_obs_squared(u, v, stim.PauliString(paulis))


def two_copy_ZZZZ(u: Simulation, v: Simulation, n: int, m: int):
    # deal with zero vs one indexing
    if not u.nqubits == v.nqubits:
        raise ValueError("Different number of qubits")
    nqubits = u.nqubits
    n, m = min(n, m), max(n, m)
    if n <= 0 or 2 * m + 2 >= nqubits:
        raise ValueError("n, m = %d, %d are not in the required range" % (n, m))

    paulis = [0] * (2 * n - 1) + [3]

    zn = [0] * (2 * n - 1) + [3]
    zn = stim.PauliString(zn + [0] * (nqubits - len(zn)))

    zm = [0] * (2 * m - 1) + [3]
    zm = stim.PauliString(zm + [0] * (nqubits - len(zm)))

    # Tr(rho ZZ rho ZZ) - Tr(rho Z_n rho Z_n) Tr(rho Z_m rho Z_m)

    return two_copy_obs_squared(u, v, zn * zm) - two_copy_obs_squared(
        u, v, zn
    ) * two_copy_obs_squared(u, v, zm)


def two_copy_endpoints(u: Simulation, v: Simulation, n: int, m: int):
    # deal with zero vs one indexing
    if not u.nqubits == v.nqubits:
        raise ValueError("Different number of qubits")
    nqubits = u.nqubits
    n, m = min(n, m), max(n, m)
    if n <= 0 or 2 * m - 1 >= nqubits:
        raise ValueError("n, m = %d, %d are not in the required range" % (n, m))

    paulis = [0] * nqubits
    paulis[2 * n - 1] = 3
    paulis[2 * m - 1] = 3
    # paulis[2 * n] = 3
    # paulis[2 * m] = 3

    return two_copy_obs_squared(u, v, stim.PauliString(paulis))


def run_single_iter(
    nqubits: int,
    p1: float,
    p2: float,
    max_depth: int = 100,
    architecture: str = "linear",
    lam: float = 0,
    pbc: bool = True,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    if p1 + p2 > 1:
        raise ValueError("Probability must sum to 1")

    if architecture not in ("linear", "brickwork", "random"):
        raise ValueError(
            "architecture should be either `linear`, `brickwork`, or `random`"
        )

    # n = int(0.25 * (nqubits // 2) + 1)
    n = 1
    # m = int(0.75 * (nqubits // 2) - 1)

    endpoints = np.zeros((max_depth + 1, nqubits // 2 - 1))
    string_W = endpoints.copy()
    string_S = endpoints.copy()
    trivial_string_W = endpoints.copy()
    trivial_string_S = endpoints.copy()
    ZZZZ = endpoints.copy()
    purity = np.zeros(max_depth + 1)

    s1, s2 = Simulation(), Simulation()
    s1.initialize(nqubits)
    s2.initialize(nqubits)

    purity[0] += two_copy_obs_squared(s1, s2)
    for m in range(nqubits // 2 - 1):
        endpoints[0, m] += two_copy_endpoints(s1, s2, n, m + 2)
        string_S[0, m] += two_copy_S(s1, s2, n, m + 2)
        string_W[0, m] += two_copy_W(s1, s2, n, m + 2)
        trivial_string_S[0, m] += two_copy_S(s1, s2, n, m + 2, True)
        trivial_string_W[0, m] += two_copy_W(s1, s2, n, m + 2, True)
        ZZZZ[0, m] += two_copy_ZZZZ(s1, s2, n, m + 2)

    for t in range(max_depth):
        if architecture == "linear":
            qubits1 = range(t % 2, nqubits, 2)
            qubits2 = range(t % 2, nqubits, 2)
            # qubits = ((2 * t + (t // (nqubits // 2) % 2)) % nqubits,)
        elif architecture == "brickwork":
            qubits1 = range(t % 3, nqubits, 3)
            qubits2 = range(t % 3, nqubits, 3)
        elif architecture == "random":
            qubits1 = (random.randint(0, nqubits - 1),)
            qubits2 = (random.randint(0, nqubits - 1),)

        for j1 in qubits1:
            r = random.random()
            if r < p2:
                s1.random_symmetric_clifford(j1, pbc)
            elif p2 <= r < p2 + p1:
                s1.xzx_measurement(j1, pbc)
                # s1.ixi_measurement(j1, pbc)
            else:
                s1.E_lambda(j1, lam, pbc)

        for j2 in qubits2:
            r = random.random()
            if r < p2:
                s2.random_symmetric_clifford(j2, pbc)
            elif p2 <= r < p2 + p1:
                s2.xzx_measurement(j2, pbc)
                # s2.ixi_measurement(j2, pbc)
            else:
                s2.E_lambda(j2, lam, pbc)

        for m in range(nqubits // 2 - 1):
            endpoints[t + 1, m] += two_copy_endpoints(s1, s2, n, m + 2)
            string_S[t + 1, m] += two_copy_S(s1, s2, n, m + 2)
            string_W[t + 1, m] += two_copy_W(s1, s2, n, m + 2)
            trivial_string_S[t + 1, m] += two_copy_S(s1, s2, n, m + 2, True)
            trivial_string_W[t + 1, m] += two_copy_W(s1, s2, n, m + 2, True)
            ZZZZ[t + 1, m] += two_copy_ZZZZ(s1, s2, n, m + 2)
        purity[t + 1] += two_copy_obs_squared(s1, s2)

    return (
        purity,
        endpoints,
        string_S,
        string_W,
        trivial_string_S,
        trivial_string_W,
        ZZZZ,
    )


def mp_run_single_iter(
    args,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    return run_single_iter(*args)


def run_two_copy_simulation(
    nqubits: int,
    p1: float,
    p2: float,
    max_depth: int = 100,
    iters: int = 1,
    architecture: str = "linear",
    lam: float = 0,
    pbc: bool = True,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:

    work_items = [(nqubits, p1, p2, max_depth, architecture, lam, pbc)] * iters
    with mp.Pool() as pool:
        output = list(pool.imap_unordered(mp_run_single_iter, work_items))

    purity, endpoints, string_S, string_W, trivial_string_S, trivial_string_W, ZZZZ = (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    for p, e, s, w, ts, tw, z in output:
        purity += p
        endpoints += e
        string_S += s
        string_W += w
        trivial_string_S += ts
        trivial_string_W += tw
        ZZZZ += z
    purity /= iters
    endpoints /= iters
    string_S /= iters
    string_W /= iters
    trivial_string_S /= iters
    trivial_string_W /= iters
    ZZZZ /= iters

    return (
        purity,
        endpoints,
        string_S,
        string_W,
        trivial_string_S,
        trivial_string_W,
        ZZZZ,
    )
