# mixed-state-phases


This repository contains code accompanying the prepint titled "Instability of steady-state mixed-state symmetry-protected topological order to strong-to-weak spontaneous symmetry breaking": https://arxiv.org/abs/2410.12900

There are two parts to the code. The first part (written by Jeet Shah) is based on Matrix Product States using ITensors in Julia. The code in the folder ./MPS/ can be used to construct the MPO corresponding to the Lindbladians mentioned in the paper, and calculate its steady states.
The code therein can also be used to calculate Renyi-1 and Renyi-2 string correlators as well as connected correlators which probe strong-to-weak spontaneous symmetry breaking.
Furthermore, we provide the code to perform the scaling collapse mentioned in the paper.

The second part of this repository is the Clifford Simulation code (written by Joseph T. Iosue) in the ./CliffordSimulation/ directory. It simulates a channel mentioned in the paper using the stim library in python.
We determine the steady state and calculate various correlators in them.
