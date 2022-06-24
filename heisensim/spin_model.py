from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import qutip as qt
from scipy.spatial.distance import pdist, squareform

from heisensim.spin_half import (
    correlator,
    single_spin_op,
    sx,
    sy,
    symmetrize_op,
    symmetrize_state,
    sz,
    up_x,
)


@dataclass()
class InteractionParams(ABC):
    normalization: Any = None

    def get_interaction(self, *args):
        int_mat = self._get_interaction(*args)
        return self.normalize(int_mat, self.normalization)

    @abstractmethod
    def _get_interaction(self, *args):
        pass

    @staticmethod
    def normalize(int_mat, normalization):
        if normalization is None:
            return int_mat

        mf = int_mat.sum(axis=1)
        if normalization == "mean":
            int_mat /= np.mean(mf)
        elif normalization == "median":
            int_mat /= np.median(mf)
        return int_mat


@dataclass()
class PowerLaw(InteractionParams):
    exponent: float = 6
    coupling: float = 1

    def _get_interaction(self, pos):
        coupling = self.coupling
        exponent = self.exponent

        dist = squareform(pdist(pos))
        np.fill_diagonal(dist, 1)
        interaction = coupling * dist ** (-exponent)
        np.fill_diagonal(interaction, 0)
        return interaction


class VanDerWaals(PowerLaw):
    def __init__(self, coupling=1, normalization=None):
        self.coupling = coupling
        self.exponent = 6
        self.normalization = normalization


class DipoleCoupling(PowerLaw):
    def __init__(self, coupling, normalization=None):
        self.coupling = coupling
        self.exponent = 3
        self.normalization = normalization

    def dipole_interaction(self, u, v):
        d = u - v
        dist = np.linalg.norm(d)
        return self.coupling * (1 - 3 * (d[2] / dist) ** 2) / dist**self.exponent

    # noinspection PyTypeChecker
    def _get_interaction(self, pos):
        return squareform(pdist(pos, self.dipole_interaction))


@dataclass()
class XYZ:
    xx: float = 1
    yy: float = 1
    zz: float = 1

    def coupling(self, model, i, j):
        return (
            self.xx * model.correlator(sx, i, j)
            + self.yy * model.correlator(sy, i, j)
            + self.zz * model.correlator(sz, i, j)
        )


class XXZ(XYZ):
    def __init__(self, delta):
        super().__init__(1, 1, delta)


class Ising(XYZ):
    def __init__(self):
        super().__init__(0, 0, 1)


class XX(XYZ):
    def __init__(self):
        super().__init__(1, 1, 0)


class SpinModel:
    def __init__(self, int_mat, int_type=XXZ(-0.6)):
        self.int_mat = int_mat
        self.int_type = int_type

    @classmethod
    def from_pos(cls, pos, int_params=VanDerWaals(), int_type=XXZ(-0.6)):
        int_mat = int_params.get_interaction(pos)
        return cls(int_mat, int_type)

    @property
    def int_mat_mf(self):
        return self.int_mat.sum(axis=1)

    @property
    def J_mean(self):
        return np.mean(self.int_mat_mf)

    @property
    def J_median(self):
        return np.median(self.int_mat_mf)

    @property
    def int_mat_nn(self):
        return self.int_mat.max(axis=1)

    @property
    def J_nn_mean(self):
        return self.int_mat_nn.mean()

    @property
    def J_nn_median(self):
        return np.median(self.int_mat_nn)

    @property
    def N(self):
        return self.int_mat.shape[0]

    def hamiltonian(self, hx=0, hy=0, hz=0):

        # external field terms
        H_field = self.hamiltonian_field(hx, hy, hz)

        # interaction terms
        H_int = self.hamiltonian_int()

        return H_int + H_field

    def hamiltonian_int(self):
        H_int = 0
        for i in range(self.N):
            for j in range(i):
                H_int += self.int_mat[i, j] * self.int_type.coupling(self, i, j)
        return H_int

    def hamiltonian_field(self, hx=0, hy=0, hz=0):
        hx = np.resize(hx, self.N)
        hy = np.resize(hy, self.N)
        hz = np.resize(hz, self.N)

        H = sum(
            hx[i] * self.single_spin_op(sx, i)
            + hy[i] * self.single_spin_op(sy, i)
            + hz[i] * self.single_spin_op(sz, i)
            for i in range(self.N)
        )
        return H  # self.symmetrize_op(H)

    def single_spin_op(self, op, n):
        return single_spin_op(op, n, self.N)

    def correlator(self, op, i, j):
        return self.single_spin_op(op, i) * self.single_spin_op(op, j)

    def get_op_list(self, op):
        return [self.single_spin_op(op, n) for n in range(self.N)]

    def product_state(self, state=up_x):
        psi0 = state.unit()
        return qt.tensor(self.N * [psi0])

    def symmetrize(self):
        return SpinModelSym(self.int_mat, self.int_type)


class SpinModelSym(SpinModel):
    def __init__(self, int_mat, int_type=XXZ(-0.6), sign=1):
        super(SpinModelSym, self).__init__(int_mat, int_type)
        self.sign = sign

    @classmethod
    def from_pos(cls, pos, int_params=VanDerWaals(), int_type=XXZ(-0.6), sign=1):
        model = super(SpinModelSym, cls).from_pos(pos, int_params, int_type)
        model.sign = sign
        return model

    def symmetrize_op(self, op):
        return symmetrize_op(op, self.sign)

    def symmetrize_state(self, state):
        return symmetrize_state(state, self.sign)

    def single_spin_op(self, op, n):
        spin_op = super(SpinModelSym, self).single_spin_op(op, n)
        return self.symmetrize_op(spin_op)

    def correlator(self, op, i, j):
        c = correlator(op, i, j, self.N)
        return self.symmetrize_op(
            c
        )  # self.single_spin_op(op, i) @ self.single_spin_op(op, j)

    def product_state(self, state=up_x):
        psi0 = state.unit()
        psi = qt.tensor(self.N * [psi0])
        return self.symmetrize_state(psi)
