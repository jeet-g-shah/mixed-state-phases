from typing import Tuple
import stim
import numpy as np
import random
from simulation import SYMMETRIC_THREE_CLIFFORD
import multiprocessing as mp


class Simulation(stim.TableauSimulator):
    def initialize(self, nqubits: int):
        self.nqubits = nqubits
        if self.nqubits % 2:
            raise ValueError("Number qubits should be even")

        # start in product state of |+> so that we're in +1 Ue sector
        self.h(*list(range(self.nqubits)))

        return self

    def initialize_random(self, nqubits: int):
        self.nqubits = nqubits
        if self.nqubits % 2:
            raise ValueError("Number qubits should be even")

        # initialize random state
        self.set_inverse_tableau(stim.Tableau.random(self.nqubits))

        # post select into Ue sector
        Ue = stim.PauliString([0, 1] * (self.nqubits // 2))
        # desired_value = False corresponds to plus
        self.postselect_observable(Ue, desired_value=False)

        return self

    def initialize_ma_state(self, nqubits: int):
        self.nqubits = nqubits
        if self.nqubits % 2:
            raise ValueError("Number qubits should be even")
        n = nqubits // 2
        z = [0] * n
        for i in range(n):
            z[i] = random.choice([-1, 1])
        for i in range(n - 1):
            if z[i] == -1:
                self.x(2 * i)
            self.h(2 * i + 1)
            if z[i] * z[i + 1] == -1:
                self.z(2 * i + 1)

        return self

    def CZ_conjugate(self):
        for i in range(self.nqubits):
            self.cz(i, (i + 1) % self.nqubits)

    def state_vector(self, *, endian: str = "little"):
        v = super().state_vector(endian=endian)
        return v.reshape(v.shape + (1,))

    def peek_Ue(self):
        return self.peek_observable_expectation(
            stim.PauliString([0, 1] * (self.nqubits // 2))
        )

    def peek_Uo(self):
        return self.peek_observable_expectation(
            stim.PauliString([1, 0] * (self.nqubits // 2))
        )

    def peek_string_order(self, n: int, m: int) -> float:
        # deal with zero vs one indexing
        n, m = min(n, m), max(n, m)
        if n <= 0 or 2 * m + 2 >= self.nqubits:
            raise ValueError("n, m = %d, %d are not in the required range" % (n, m))
        paulis = [0] * (2 * n - 2) + [3]
        for j in range(n, m):
            paulis.append(1)
            paulis.append(0)
        paulis.append(1)
        paulis.append(3)
        paulis += [0] * (self.nqubits - len(paulis))
        return self.peek_observable_expectation(stim.PauliString(paulis))

    def peek_trivial_string_order(self, n: int, m: int) -> float:
        # deal with zero vs one indexing
        n, m = min(n, m), max(n, m)
        if n <= 0 or 2 * m + 2 >= self.nqubits:
            raise ValueError("n, m = %d, %d are not in the required range" % (n, m))
        paulis = [0] * (2 * n - 1)
        for j in range(n, m):
            paulis.append(1)
            paulis.append(0)
        paulis.append(1)
        paulis += [0] * (self.nqubits - len(paulis) + 1)
        return self.peek_observable_expectation(stim.PauliString(paulis))

    def peek_zz(self, n: int, m: int) -> float:
        # Z and Z on even sites (so in the python indexing, odd sites)
        n, m = min(n, m), max(n, m)
        x, y = 2 * n - 1, 2 * m + 1
        if x < 0 or y >= self.nqubits:
            raise ValueError("n, m = %d, %d are not in the required range" % (n, m))
        paulis = [0] * self.nqubits
        paulis[x] = 3
        paulis[y] = 3
        # return self.peek_observable_expectation(stim.PauliString(paulis))
        return self.peek_observable_expectation(stim.PauliString(paulis)) - self.peek_z(
            x
        ) * self.peek_z(y)

    def E(self, j: int, pbc: bool = True) -> bool:
        # we are using 0 index. So j being odd corresponds to E_even
        # and j being even corresponds to E_odd
        m = False

        if j % 2:  # j is odd in 0 index so even in 1 index
            # apply E_even
            if j < self.nqubits - 1:
                pauli = [0] * (j - 1) + [3, 1, 3] + [0] * (self.nqubits - j - 2)

                ## wait shouldn't it be this? But this doesn't work
                # pauli = [0] * (j - 2) + [3, 1, 3] + [0] * (self.nqubits - j - 1)

                # True corresponds to a - measurement
                if m := self.measure_observable(stim.PauliString(pauli)):
                    self.x(j + 1)
            elif pbc:  # pbc
                pauli = [3] + [0] * (self.nqubits - 3) + [3, 1]
                # True corresponds to a - measurement
                if m := self.measure_observable(stim.PauliString(pauli)):
                    self.x(0)

        elif (
            pbc or 1 <= j <= self.nqubits - 2
        ):  # j is even in 0 index so odd in 1 index
            # apply E_odd
            m = self.measure(j)
            self.z((j - 1) % self.nqubits)
            self.x(j)
            self.z((j + 1) % self.nqubits)

        return m

    def E_CZconjugated(self, j: int, pbc: bool = True) -> bool:

        ## Sanity check
        # should be the same as:
        # self.CZ_conjugate()
        # m = self.E(j, pbc)
        # self.CZ_conjugate()
        # return m

        # we are using 0 index. So j being odd corresponds to E_even
        # and j being even corresponds to E_odd
        m = False

        if j % 2:  # j is odd in 0 index so even in 1 index
            # apply E_even
            if pbc or j + 2 < self.nqubits:
                pauli = [0] * j + [1] + [0] * (self.nqubits - j - 1)

                ## wait shouldn't it be this? But this doesn't work
                # pauli = [0] * (j - 1) + [1] + [0] * (self.nqubits - j)

                # True corresponds to a - measurement
                if m := self.measure_observable(stim.PauliString(pauli)):
                    self.z(j)
                    self.x((j + 1) % self.nqubits)
                    self.z((j + 2) % self.nqubits)

        else:  # j is even in 0 index so odd in 1 index
            # apply E_odd
            m = self.measure(j)
            self.x(j)

        return m

    def E_lambda(self, j: int, lam: float, pbc: bool = True) -> bool:
        if random.random() < lam:
            return self.E_CZconjugated(j, pbc)
        return self.E(j, pbc)

    def xzx_measurement(self, j: int, pbc: bool = True) -> bool:
        if j == 0 and pbc:
            pauli = [3, 1] + [0] * (self.nqubits - 3) + [1]
        elif j == self.nqubits - 1 and pbc:
            pauli = [1] + [0] * (self.nqubits - 3) + [1, 3]
        else:
            pauli = [0] * (j - 1) + [1, 3, 1] + [0] * (self.nqubits - j - 2)

        return self.measure_observable(stim.PauliString(pauli))

    def ixi_measurement(self, j: int, pbc: bool = True) -> bool:
        pauli = [0] * j + [1] + [0] * (self.nqubits - j - 1)
        return self.measure_observable(stim.PauliString(pauli))

    def random_symmetric_clifford(self, j: int, pbc: bool = True) -> None:
        if pbc or 1 <= j < self.nqubits - 1:
            t = random.choice(SYMMETRIC_THREE_CLIFFORD)
            self.do_tableau(t, [(j - 1) % self.nqubits, j, (j + 1) % self.nqubits])


def run_single_iter(
    nqubits: int,
    p1: float,
    p2: float,
    max_depth: int = 100,
    architecture: str = "linear",
    lam: float = 0,
    pbc: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    if p1 + p2 > 1:
        raise ValueError("Probability must sum to 1")

    if architecture not in ("linear", "brickwork", "random"):
        raise ValueError(
            "architecture should be either `linear`, `brickwork`, or `random`"
        )

    n = int(0.25 * (nqubits // 2) + 1)
    m = int(0.75 * (nqubits // 2) - 1)

    string = np.zeros(max_depth + 1)
    trivial_string = np.zeros(max_depth + 1)

    s = Simulation()
    s.initialize(nqubits)
    string[0] += s.peek_string_order(n, m)
    trivial_string[0] += s.peek_trivial_string_order(n, m)

    for t in range(max_depth):
        if architecture == "linear":
            qubits = range(t % 2, nqubits, 2)
            # qubits = ((2 * t + (t // (nqubits // 2) % 2)) % nqubits,)
        elif architecture == "brickwork":
            qubits = range(t % 3, nqubits, 3)
        elif architecture == "random":
            qubits = (random.randint(0, nqubits - 1),)

        for j in qubits:
            r = random.random()
            if r < p2:
                s.random_symmetric_clifford(j, pbc)
            elif p2 <= r < p2 + p1:
                s.xzx_measurement(j, pbc)
                # s.ixi_measurement(j, pbc)
            else:
                s.E_lambda(j, lam, pbc)

        string[t + 1] += s.peek_string_order(n, m)
        trivial_string[t + 1] += s.peek_trivial_string_order(n, m)

    return string, trivial_string


## to use mp.Pool for parallization, need to have a globally defined function
def mp_run_single_iter(args) -> Tuple[np.ndarray, np.ndarray]:
    return run_single_iter(*args)


def run_simulation(
    nqubits: int,
    p1: float,
    p2: float,
    max_depth: int = 100,
    iters: int = 1,
    architecture: str = "linear",
    lam: float = 0,
    pbc: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:

    work_items = [(nqubits, p1, p2, max_depth, architecture, lam, pbc)] * iters
    with mp.Pool() as pool:
        output = list(pool.imap_unordered(mp_run_single_iter, work_items))

    string, trivial_string = 0, 0
    for s, ts in output:
        string += s
        trivial_string += ts
    string /= iters
    trivial_string /= iters

    return string, trivial_string
