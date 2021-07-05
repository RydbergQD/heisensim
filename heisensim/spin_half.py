import qutip as qt
from qutip.fastsparse import csr_matrix
import numpy as np

sx = qt.sigmax() / 2
sy = qt.sigmay() / 2
sz = qt.sigmaz() / 2
si = qt.qeye(2)

up = qt.basis(2, 0)
down = qt.basis(2, 1)
up_x = 1/np.sqrt(2) * (up + down)
up_y = 1/np.sqrt(2) * (up - down)


def bloch_state(theta=np.pi/2, phi=0):
    return np.cos(theta/2) * up + np.exp(-1j*phi) * np.sin(theta/2) * down


def symmetrize_state(state, sign=1):
    if isinstance(state, qt.Qobj):
        state = state.data.toarray()[:, 0]
    dim = state.shape[0]
    new_dim = dim // 2
    state_l = state[:new_dim]
    state_r = state[-1:new_dim - 1:-1]
    state_sym = 1 / np.sqrt(2) * (state_l + sign * state_r)
    if np.isreal(state_sym).all():
        return np.real(state_sym)
    return state_sym


def symmetrize_op(op, sign=1):
    """
    symmetrize an operator with respect to parity symmetry.
    :param op: operator, either matrix or qobj.
    :param sign: Whether to obtain symmetric or antisymmetric part.
     Choose -1 for antisymmetric.
    :return: (anti-)symmetrized operator
    """
    if isinstance(op, qt.Qobj):
        op = op.data
    dim = op.shape[0]
    new_dim = dim // 2

    H_ul = op[0:new_dim, 0:new_dim]
    H_ur = op[0:new_dim, -1:(new_dim - 1):-1]
    H_ll = op[-1:(new_dim - 1):-1, 0:new_dim]
    H_lr = op[-1:(new_dim - 1):-1, -1:(new_dim - 1):-1]

    op_sym = 0.5 * (H_ul + H_lr + sign * (H_ll + H_ur))
    if np.isreal(op_sym.data).all():
        op_sym.data = op_sym.data.real
    return csr_matrix(op_sym)  # .toarray()


def single_spin_op(op, n, N):
    op_list = list(si for _ in range(N))
    op_list[n] = op
    return qt.tensor(op_list)


def correlator(op, i, j, N):
    return single_spin_op(op, i, N) * single_spin_op(op, j, N)


def get_op_list(op, N):
    return [single_spin_op(op, n, N) for n in range(N)]


def expect(op, state):
    """
    Calculate expectation value <state|op|state>.
    :param op: (..., M, M) ndarray
    operator
    :param state:  (..., M, M) ndarray
    The column ``v[:, i]`` is a single state.
    :return: (..., M) ndarray
    the value v[i] corresponds to the expectation value corresponding to the state state[:, i]
    """
    if state.ndim == 2:
        return np.einsum('ij, ji -> i', state.T, op @ state)
    else:
        return state @ op @ state


class TimeEvolution:
    def __init__(self, psi_0, op, e_vals, e_states):
        self.C = psi_0 @ e_states
        energy_differences = np.tile(e_vals, (e_vals.shape[0], 1))
        self.energy_diff_array = energy_differences - energy_differences.T
        self.eev_including_off_diagonal = e_states.T @ op @ e_states
        self.phases = np.zeros_like(self.energy_diff_array, dtype=np.complex128)
        self.einsum_path = np.einsum_path(
            'a, b, ab, ab -> ',
            self.C, self.C, self.phases, self.eev_including_off_diagonal, optimize='optimal'
        )[0]

    def __call__(self, t):
        np.exp(- 1j * self.energy_diff_array * t, out=self.phases)
        return np.einsum(
            'a, b, ab, ab -> ',
            self.C, self.C, self.phases, self.eev_including_off_diagonal,
            optimize=self.einsum_path
        )
