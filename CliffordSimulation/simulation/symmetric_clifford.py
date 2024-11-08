import stim
import itertools


THREE_PAULIS = [
    stim.PauliString("".join(p)) for p in itertools.product("IXYZ", repeat=3)
][1:]
THREE_PAULIS.extend([-x for x in THREE_PAULIS])

xix = stim.PauliString("XIX")
ixi = stim.PauliString("IXI")
# for symmetry, we need x1*x3 = xix, x2 = ixi

iii = stim.PauliString("III")

x2 = ixi
x1s = filter(
    lambda p: p != x2
    and p.commutes(x2)
    and xix.commutes(p)
    and p * xix * p * xix == iii
    and x2.commutes(p * xix),
    THREE_PAULIS,
)

SYMMETRIC_THREE_CLIFFORD = []

for x1 in x1s:
    x3 = x1 * xix

    for z1 in filter(
        lambda p: p.commutes(x2) and p.commutes(x3) and not p.commutes(x1),
        THREE_PAULIS,
    ):
        for z2 in filter(
            lambda p: p.commutes(x1)
            and p.commutes(x3)
            and not p.commutes(x2)
            and p.commutes(z1),
            THREE_PAULIS,
        ):
            for z3 in filter(
                lambda p: p.commutes(x1)
                and p.commutes(x2)
                and not p.commutes(x3)
                and p.commutes(z1)
                and p.commutes(z2),
                THREE_PAULIS,
            ):
                SYMMETRIC_THREE_CLIFFORD.append(
                    stim.Tableau.from_conjugated_generators(
                        xs=[x1, x2, x3], zs=[z1, z2, z3]
                    )
                )
