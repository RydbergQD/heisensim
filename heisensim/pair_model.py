from dataclasses import dataclass, field
import numpy as np
from functools import cached_property
from scipy.spatial.distance import squareform, pdist
from scipy.optimize import root
from tqdm.auto import tqdm

from . import (
    get_energy_diff,
    DipoleCoupling,
    XYZ,
    InteractionParams,
    get_canonical_ensemble,
)

from scipy.optimize import minimize_scalar


@dataclass
class Clusterer:
    int_range: InteractionParams
    int_type: XYZ
    positions: np.ndarray

    def __post_init__(self):
        assert self.int_type.xx == self.int_type.yy

    @property
    def Delta(self):
        return self.int_type.zz / self.int_type.xx

    @cached_property
    def dist(self):
        dist = squareform(pdist(self.positions))
        np.fill_diagonal(dist, np.nan)
        return dist

    @cached_property
    def inter_mat(self):
        return self.int_type.xx / 4 * self.int_range.get_interaction(self.positions)

    def get_pair_positions(self, pairs):
        return np.array(
            [
                [self.positions[nn_pair[0]], self.positions[nn_pair[1]]]
                for nn_pair in pairs
            ]
        )

    def get_pair_interactions(self, pairs):
        return np.array([self.inter_mat[pair[0], pair[1]] for pair in pairs])


@dataclass
class PairClusterer(Clusterer):
    def find_pairs(self):
        pairs = []
        inter_mat = np.abs(self.inter_mat.copy())
        for _ in tqdm(range(inter_mat.shape[0] // 2)):
            nn_pair = np.unravel_index(np.nanargmax(inter_mat), inter_mat.shape)
            pairs.append(nn_pair)
            inter_mat[nn_pair[0], :] = np.nan
            inter_mat[:, nn_pair[0]] = np.nan
            inter_mat[nn_pair[1], :] = np.nan
            inter_mat[:, nn_pair[1]] = np.nan
        return pairs

    def all_interactions_between_pairs(self, pair1, pair2):
        try:
            if np.all(pair1 == pair2):
                return [0] * 4
            return [self.inter_mat[p1, p2] for p1 in pair1 for p2 in pair2]
        except:
            print(pair1, pair2)
            raise KeyError()

    def get_interactions_between_pairs(self, pairs, reduce=np.mean, progress=False):
        if progress:
            iterator = tqdm(pairs)
        else:
            iterator = pairs
        J_inter_list = np.array(
            [
                [self.all_interactions_between_pairs(p1, p2) for p1 in pairs]
                for p2 in iterator
            ]
        )
        return reduce(J_inter_list, axis=-1)


@dataclass
class MACEClusterer(Clusterer):
    def find_pairs(self):
        inter_mat = self.inter_mat
        nn = np.nanargmax(np.abs(inter_mat), axis=1)
        pairs = np.array([[i, n] for i, n in enumerate(nn)])
        return pairs

    def get_interactions_between_pairs(self, pairs, reduce=np.mean, progress=False):
        inter_mat = self.inter_mat.copy()
        for i, n in pairs:
            inter_mat[i, n] = 0
            inter_mat[n, i] = 0
        return inter_mat


@dataclass
class SinglePair:
    J: float
    Delta: float

    @classmethod
    def from_pair(cls, pos1, pos2, int_range=DipoleCoupling(2 * np.pi), int_type=XYZ()):
        J = int_type.xx / 4 * int_range.get_interaction_pair(pos1, pos2)
        Delta = int_type.zz / int_type.xx
        return cls(J=J, Delta=Delta)

    @property
    def j(self):
        return self.J * (self.Delta - 1)

    def get_E0(self, h=0):
        return h + self.J

    def get_evals(self, h=0, sign=1):
        return self.J - sign * np.sqrt(h**2 + self.j**2)

    def get_eev(self, h=0, sign=1):
        return -sign * h / (2 * np.sqrt(h**2 + self.j**2))

    def get_eon(self, h=0, sign=1):
        return 0.5 + sign * h / (2 * np.sqrt(h**2 + self.j**2))

    def get_diagonal(self, h=0):
        return np.true_divide(
            h**2,
            2 * (h**2 + self.j**2),
            out=np.zeros(np.shape(h)),
            where=(h != 0),
        )

    def get_energy_diff_from_beta(self, beta, h):
        e_diff = (
            -np.sqrt(h**2 + self.j**2)
            * np.tanh(np.sqrt(h**2 + self.j**2) * beta)
            - h
        )
        return abs(e_diff.sum())

    def get_canonical_from_beta(self, beta, h):
        return -(h * np.tanh(np.sqrt(h**2 + self.j**2) * beta)) / (
            2 * np.sqrt(h**2 + self.j**2)
        )

    def get_canonical_ensemble(self, h=0, beta_0=0):
        ev = np.array([self.get_evals(h=h, sign=-1), self.get_evals(h=h, sign=1)])
        eevs = np.array([self.get_eev(h=h, sign=-1), self.get_eev(h=h, sign=1)])
        E0 = self.get_E0(h=h)
        return np.dot(get_canonical_ensemble(ev, E0, beta_0=beta_0), eevs)

    def time_evolution(self, t, h):
        return self.j**2 * np.cos(2 * np.sqrt(h**2 + self.j**2) * t)/(2 * (h**2 + self.j**2)) + self.get_diagonal(h)

    def time_differential(self, t, h):
        return -self.j**2 * np.sin(2 * np.sqrt(h**2 + self.j**2) * t)


@dataclass
class PairEnsemble(SinglePair):
    @classmethod
    def from_pairs(cls, clusterer: Clusterer):
        pairs = clusterer.find_pairs()
        J_list = clusterer.get_pair_interactions(pairs)
        return cls(J=J_list, Delta=0)

    def get_diagonal(self, h=0):
        return h**2 / (2 * (h**2 + self.j**2))

    def get_E0(self, h=0):
        return np.sum(super().get_E0(h=h))

    def get_beta(self, h, beta_0=0, J_median=1):
        return minimize_scalar(
            self.get_energy_diff_from_beta,
            bracket=[-0.001 * J_median, 0.001 * J_median],
            args=(h,),
        ).x

    def get_canonical_ensemble(self, h=0, beta_0=0):
        if np.all(h == 0):
            return 0
        beta = self.get_beta(h, beta_0=beta_0)
        return self.get_canonical_from_beta(beta, h)


@dataclass
class PairEnsembleMeanfield(PairEnsemble):
    J_inter_list: np.ndarray
    N: int = field(init=False)

    @classmethod
    def from_pairs(cls, clusterer: Clusterer):
        pairs = clusterer.find_pairs()
        J_list = clusterer.get_pair_interactions(pairs)
        J_inter_list = clusterer.get_interactions_between_pairs(pairs)
        return cls(J=J_list, Delta=0, J_inter_list=J_inter_list)

    def __post_init__(self):
        self.N = len(self.J)
        assert np.shape(self.J_inter_list)[0] == self.N
        assert np.shape(self.J_inter_list)[1] == self.N

    def _get_diagonal(self, s_x, h):
        return super().get_diagonal(h + self.J_inter_list @ s_x) - s_x

    def get_diagonal(self, h, maxiter=100, method="Krylov", xtol=1e-8):
        init = 0  # np.sign(h)
        return root(
            self._get_diagonal,
            init * np.ones(self.N),
            args=(h,),
            method=method,
            options=dict(maxiter=maxiter, xtol=xtol),
        ).x

    def _get_canonical(self, s_x, h):
        return super().get_canonical_ensemble(h + self.J_inter_list @ s_x) - s_x

    def get_canonical_ensemble(self, h, maxiter=100, method="Krylov", xtol=1e-8):
        if h == 0:
            return np.zeros(self.N)
        init = 0  # np.sign(h)
        return root(
            self._get_canonical,
            init * np.ones(self.N),
            args=(h,),
            method=method,
            options=dict(maxiter=maxiter, xtol=xtol),
        ).x
