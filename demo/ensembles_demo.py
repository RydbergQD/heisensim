import h5py
from pathlib import Path
import argparse
import numpy as np
import xarray as xr
from pathlib import Path
from scipy.spatial import cKDTree

import heisensim as sim

parser = argparse.ArgumentParser(description='Calculate ensemble expectation values.')
parser.add_argument('--spin_number', type=int, default=10,
                    help='number of spins. Hilbert space has dimension 2**N')
parser.add_argument('--field', "-f", type=float, default=0.0,
                    help="External field, vary between -10 and 10")

args = parser.parse_args()
h = args.field
N = args.spin_number

path = Path.cwd()
with h5py.File(path / "positions" / 'positions.jld2', 'r') as file:
    positions = file["cusp_21_11_2020"]["pos"][:]
positions = np.array(positions)
tree = cKDTree(data=positions)
spins = positions.shape[0]


h_list = [h]
disorder_array = np.arange(spins)


for i in disorder_array:
    pos_i = positions[i]
    _, pos_indices = tree.query(pos_i, N)
    pos = positions[pos_indices]
    model = sim.SpinModelSym.from_pos(pos, int_params=sim.DipoleCoupling(1, normalization='mean'),
                                      int_type=sim.XX())
    magn = 1 / N * sum(model.get_op_list(sim.sx))
    H_int = model.hamiltonian()
    J_median = np.median(model.int_mat.sum(axis=0))
    psi_0 = model.product_state()

    H = H_int + model.hamiltonian_field(hx=h)
    e_vals, e_states = np.linalg.eigh(H.toarray())
    for spin, op in enumerate(model.get_op_list(sim.sx)):
        eev = sim.expect(op, e_states)

    eon = psi_0 @ e_states
    E_0 = sim.expect(H, psi_0)
    delta_E_0 = np.sqrt(sim.expect(H @ H, psi_0) - E_0 ** 2)

    break
