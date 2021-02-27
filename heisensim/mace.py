from scipy.spatial.ckdtree import cKDTree
from heisensim.spin_model import SpinModelSym, XX, DipoleCoupling
from heisensim.spin_half import expect, sx
from heisensim.thermalization import diagonal_ensemble, micro_ensemble, canonical_ensemble
import numpy as np
import pandas as pd


def pos_for_mace(positions, spin_number, cluster_size):
    tree = cKDTree(data=positions)
    pos_i = positions[spin_number]
    _, pos_indices = tree.query(pos_i, cluster_size)
    pos = positions[pos_indices]
    return pos


def single_mace_run(positions, result_list, args):
    spin, h = args
    pos = pos_for_mace(positions, spin, cluster_size)
    model = SpinModelSym.from_pos(pos, int_params=DipoleCoupling(1200, normalization=None),
                                      int_type=XX())
    H_int = model.hamiltonian()
    psi_0 = model.product_state()
    H = H_int + model.hamiltonian_field(hx=h)
    e_vals, e_states = np.linalg.eigh(H.toarray())
    eon = psi_0 @ e_states
    E_0 = expect(H, psi_0)
    delta_E_0 = np.sqrt(expect(H @ H, psi_0) - E_0 ** 2)
    op = model.single_spin_op(sx, 0)
    eev = expect(op, e_states)
    diag = eev @ diagonal_ensemble(eon)
    micro = eev @ micro_ensemble(e_vals, E_0, delta_E=0.05)
    canonical = eev @ canonical_ensemble(e_vals, E_0)
    result_list.append(
        pd.DataFrame([[E_0, delta_E_0, diag, micro, canonical, spin, h]],
                     columns=["E_0", "delta_E_0", "diag", "micro", "canonical", "spin", "h"])
    )
    print(result_list)
