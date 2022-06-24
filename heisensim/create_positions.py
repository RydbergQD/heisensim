import numpy as np
import xarray as xr
import heisensim as sim
from pathlib import Path


def radius_from_packing(packing_density=0.74, N=12, r_bl=1):
    return r_bl/2 * ((N/packing_density)**(1/3))


def create_positions(path, N, disorder_realizations=100):
    r_bl_list = np.round(np.arange(0.2, 0.95, 0.05), 2)
    disorder_realization_list = np.arange(disorder_realizations)

    positions = xr.DataArray(
        np.zeros((len(r_bl_list), len(disorder_realization_list), N, 3), dtype='float'),
        dims=['r_bl', 'disorder_realization', 'particle', 'xyz'],
        coords={
            'r_bl': r_bl_list,
            'disorder_realization': disorder_realization_list,
            'particle': np.arange(N),
            'xyz': ['x', 'y', 'z']
        }
    )

    for r_bl in r_bl_list:
        sphere = sim.Sphere(r_bl=r_bl, radius=radius_from_packing(N=N), max_iter=int(1e5))
        for disorder_realization in disorder_realization_list:
            pos = sphere.sample_positions(N)
            positions.loc[r_bl, disorder_realization] = pos

    positions.to_netcdf(path / "positions_{}.nc".format(N))
    return positions


if __name__ == '__main__':
    path = Path.cwd().parent / "positions"
    create_positions(path, 6)
