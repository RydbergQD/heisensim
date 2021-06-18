from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
import numpy as np
import xarray as xr
from functools import cached_property


def get_weights_canonical(ev, beta):
    log_sum_exp = logsumexp(-beta * ev)
    weights = np.exp(-beta * ev - log_sum_exp)
    return weights


def get_energy_diff(ev, beta, E_0):
    weights = get_weights_canonical(ev, beta)
    return np.abs(weights @ ev - E_0)


def get_beta(ev, E_0, beta_0=0):
    return minimize_scalar(lambda beta: get_energy_diff(np.array(ev), beta, E_0)).x


def get_canonical_ensemble(ev, E_0, beta_0=0):
    beta = get_beta(ev, E_0, beta_0=beta_0)
    return get_weights_canonical(beta.x, ev)


def get_micro_ensemble(ev, E_0, delta_E=10):
    micro = (ev < E_0 + delta_E) & (ev > E_0 - delta_E)
    if micro.sum() > 0:
        return micro / micro.sum()
    else:
        micro = 0 * ev
        micro[np.argmin(np.array((ev - E_0) ** 2))] = 1
        return micro


def diagonal_ensemble(eon):
    return abs(eon) ** 2


class ThermalEnsemble:
    state_dim = "state"

    def __init__(self, xarray_obj, beta_0=0, delta_E=10):
        self._obj = xarray_obj
        self.beta_0 = beta_0
        self.delta_E = delta_E

        self.outer_dim = self.extract_outer_dim()

    def extract_outer_dim(self):
        outer_dims = list(set(self._obj.dims.keys()) - {self.state_dim})
        if len(outer_dims) != 1:
            raise Warning(
                "All functions that depend on xarray groupby capabilities won't work"
            )
        return outer_dims[0]

    @cached_property
    def E_0(self):
        """Return the energy of the initial state."""
        return xr.dot(self._obj.e_vals, self._obj.eon, dims=self.state_dim)

    @cached_property
    def E_fluctuations(self):
        """Return the energy fluctuations of the initial state."""
        E_fluctuations = xr.dot(
            self._obj.e_vals ** 2, self._obj.eon, dims=self.state_dim
        )
        return np.sqrt(E_fluctuations - self.E_0 ** 2)

    @cached_property
    def beta(self):
        """Return the temperature of the most propable canonical ensemble state."""
        beta = xr.apply_ufunc(
            get_beta,
            self._obj.e_vals.groupby(self.outer_dim),
            self.E_0.groupby(self.outer_dim),
            input_core_dims=([self.state_dim], []),
            kwargs={"beta_0": self.beta_0},
        )
        return beta

    @cached_property
    def canonical(self):
        beta = self.beta
        weights_canonical = xr.apply_ufunc(
            get_weights_canonical,
            self._obj.e_vals.groupby(self.outer_dim),
            beta.groupby(self.outer_dim),
        )
        return xr.dot(weights_canonical, self._obj.eev, dims=self.state_dim)

    @cached_property
    def diagonal(self):
        return xr.dot(self._obj.eon, self._obj.eev, dims=self.state_dim)

    @cached_property
    def micro(self):
        weights_micro = xr.apply_ufunc(
            get_micro_ensemble,
            self._obj.e_vals.groupby(self.outer_dim),
            self.E_0.groupby(self.outer_dim),
            kwargs={"delta_E": self.delta_E},
        )
        return xr.dot(weights_micro, self._obj.eev, dims=self.state_dim)

    def get_summary(self):
        return xr.Dataset(
            {
                "E_0": self.E_0,
                "E_fluctuations": self.E_fluctuations,
                "beta": self.beta,
                "canonical": self.canonical,
                "diagonal": self.diagonal,
                "micro": self.micro,
            }
        )
