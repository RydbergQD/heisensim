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


def evaluate_steady_state(results):
    E_0 = xr.dot(results.e_vals, results.eon, dims="state")
    E_fluctuations = xr.dot(results.e_vals ** 2, results.eon, dims="state")
    E_fluctuations = np.sqrt(E_fluctuations - E_0 ** 2)

    beta = get_beta_xr(results, E_0)
    weights_canonical = xr.apply_ufunc(
        get_weights_canonical, results.e_vals.groupby("pos_i"), beta.groupby("pos_i")
    )
    canonical = xr.dot(weights_canonical, results.eev, dims="state")
    diagonal = xr.dot(results.eev, results.eon, dims="state")
    weights_micro = xr.apply_ufunc(
        get_micro_ensemble,
        results.e_vals.groupby("pos_i"),
        kwargs={"E_0": 0, "delta_E": 0.1},
    )
    micro = xr.dot(weights_micro, results.eev, dims="state")

    return xr.Dataset(
        {
            "E_0": E_0,
            "E_fluctuations": E_fluctuations,
            "beta": beta,
            "canonical": canonical,
            "diagonal": diagonal,
            "micro": micro,
        }
    )

@xr.register_dataset_accessor("ensembles")
class ThermalEnsemble:
    state_dim = 'state'
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        outer_dims = list(set(cusp.dims.keys()) - {'state'})
        if len(outer_dims) != 1:
            raise Warning("All functions that depend on xarray groupby capabilities won't work")
        self.outer_dim = outer_dims[0]

    @cached_property
    def E_0(self):
        """Return the energy of the initial state."""
        return xr.dot(self._obj.e_vals, self._obj.eon, dims="state")
    
    @cached_property
    def E_fluctuations(self):
        """Return the energy fluctuations of the initial state."""
        E_fluctuations = xr.dot(results.e_vals ** 2, results.eon, dims="state")
        return np.sqrt(E_fluctuations - E_0 ** 2)

    def get_beta(self):
        """Return the temperature of the most propable canonical ensemble state."""
        beta = xr.apply_ufunc(
            sim.get_beta,
            results.e_vals.groupby(self.outer_dim),
            E_0.groupby(self.outer_dim),
            input_core_dims=([self.state_dim], []),
        )
        return beta
    

