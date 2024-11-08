import matplotlib.pyplot as plt
from matplotlib import rc

plt.rcParams["text.usetex"] = True
font = {
    "family": "sf",
    # "weight" : "bold",
    "size": 16,
}
rc("font", **font)

from simulation.heatmap import heatmap
from simulation.symmetric_clifford import SYMMETRIC_THREE_CLIFFORD
from simulation.run_simulation import Simulation, run_simulation
