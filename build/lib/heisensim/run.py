import argparse
import numpy as np
import xarray as xr
from pathlib import Path

import heisensim as sim


def radius_from_packing(packing_density=0.74, N=12, r_bl=1):
    return r_bl / 2 * ((N / packing_density) ** (1 / 3))


parser = argparse.ArgumentParser(description='Calculate ensemble expectation values.')
parser.add_argument('--spin_number', type=int, default=10,
                    help='number of spins. Hilbert space has dimension 2**N')
parser.add_argument('--blockade_radius', "-r_bl", type=float, default=0.5,
                    help="Blockade radius, vary between 0.2 and 0.95")
parser.add_argument('--field', "-f", type=float, default=0.0,
                    help="External field, vary between -10 and 10")

args = parser.parse_args()
N = args.spin_number
r_bl = args.blockade_radius
h = args.field

cwd = path = Path.cwd()
positions = xr.load_dataarray(cwd.parent / "positions" / "positions_{}.nc".format(N))

h_list = [h]
empty_array = np.zeros((1, 50, 1, 2 ** (N - 1)), dtype=np.float64)
empty_array_eev = np.zeros((1, 50, 1, 2 ** (N - 1), N), dtype=np.float64)
empty_array_E_0 = np.zeros((1, 50, 1), dtype=np.float64)
disorder_array = np.arange(50)

simulation_results = xr.Dataset(
    {
        'e_vals': (['r_bl', 'disorder_realization', 'h', 'eigen_state'], empty_array.copy()),
        'eev': (['r_bl', 'disorder_realization', 'h', 'eigen_state', 'spin'], empty_array_eev.copy()),
        'eon': (['r_bl', 'disorder_realization', 'h', 'eigen_state'], empty_array.copy()),
        'E_0': (['r_bl', 'disorder_realization', 'h'], empty_array_E_0.copy()),
        'delta_E_0': (['r_bl', 'disorder_realization', 'h'], empty_array_E_0.copy())
    },
    coords={
        'r_bl': [r_bl],
        'disorder_realization': disorder_array,
        'h': [h],
        'eigen_state': np.arange(2 ** (N - 1)),
        'spin': np.arange(N)
    }
)

for i in disorder_array:
    pos = positions.loc[r_bl, i]
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
        simulation_results.eev.loc[r_bl, i, h, :, spin] = eev

    eon = psi_0 @ e_states
    E_0 = sim.expect(H, psi_0)
    delta_E_0 = np.sqrt(sim.expect(H @ H, psi_0) - E_0 ** 2)

    simulation_results.e_vals.loc[r_bl, i, h] = e_vals
    simulation_results.eon.loc[r_bl, i, h] = eon
    simulation_results.E_0.loc[r_bl, i, h] = E_0
    simulation_results.delta_E_0.loc[r_bl, i, h] = delta_E_0

save_path = Path("/scratch/users/jfranz/Heisenberg")
simulation_results.to_netcdf(
    save_path /  "run_N-{}_rbl-{}_h-{}.nc".format(
        N, r_bl, h)
)
