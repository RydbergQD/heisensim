from functools import partial
from multiprocessing import Manager, Pool
from itertools import product

import h5py
import argparse
import numpy as np
from pathlib import Path
import pandas as pd

import heisensim as sim


def single_mace_run(cluster_size, positions, result_list, args):
    spin, h = args
    pos = sim.pos_for_mace(positions, spin, cluster_size)
    model = sim.SpinModelSym.from_pos(pos, int_params=sim.DipoleCoupling(1200, normalization=None),
                                      int_type=sim.XX())
    H_int = model.hamiltonian()
    psi_0 = model.product_state()
    H = H_int + model.hamiltonian_field(hx=h)
    e_vals, e_states = np.linalg.eigh(H.toarray())
    eon = psi_0 @ e_states
    E_0 = sim.expect(H, psi_0)
    delta_E_0 = np.sqrt(sim.expect(H @ H, psi_0) - E_0 ** 2)
    op = model.single_spin_op(sim.sx, 0)
    eev = sim.expect(op, e_states)
    diag = eev @ sim.diagonal_ensemble(eon)
    micro = eev @ sim.micro_ensemble(e_vals, E_0, delta_E=0.05)
    canonical = eev @ sim.canonical_ensemble(e_vals, E_0)
    result_list.append(
        pd.DataFrame([[E_0, delta_E_0, diag, micro, canonical, spin, h]],
                     columns=["E_0", "delta_E_0", "diag", "micro", "canonical", "spin", "h"])
    )


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--fields",  # name on the CLI - drop the `--` for positional/required parameters
        nargs="*",  # 0 or more values expected => creates a list
        type=float,
        default=[-0.1, 0, 0.1],  # default if nothing is provided
        help="External field, vary between -10 and 10"
    )
    CLI.add_argument('--cluster_size', type=int, default=10,
                     help='number of spins. Hilbert space has dimension 2**N')

    # parse the command line
    args = CLI.parse_args()
    # access CLI options
    h_list = args.fields
    print(h_list)
    cluster_size = args.cluster_size

    path = Path.cwd()
    with h5py.File(path / "positions" / 'positions.jld2', 'r') as file:
        positions = file["cusp_21_11_2020"]["pos"][:]
    positions = np.array(positions)
    spins = positions.shape[0]
    magn = 1 / cluster_size * sum(sim.get_op_list(sim.sx, cluster_size))

    dfs_list = Manager().list()
    pool = Pool()
    res = pool.map_async(partial(single_mace_run, cluster_size, positions, dfs_list),
                         product(np.arange(3), h_list))
    res.wait()
    dfs = pd.concat(dfs_list, ignore_index=True)  # the final result
    print(dfs)

    # save_path = path / "simulation_results"  #
    # save_path = Path("/scratch/users/jfranz/Heisenberg")
    # results.to_csv(
    #     save_path / "mace_cluster-{}_h-{}.csv".format(
    #         cluster_size, h)
    # )
