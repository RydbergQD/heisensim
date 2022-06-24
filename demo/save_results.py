
coords = {
    'disorder_realization': disorder_array,
    'h': h_list,
    'eigen_state': np.arange(2 ** (N - 1)),
    'spin': np.arange(N)
}

simulation_results = xr.Dataset(
    {
        'e_vals': sim.coords_tuple(['disorder_realization', 'h', 'eigen_state'], coords),
        'eev': sim.coords_tuple(['disorder_realization', 'h', 'eigen_state', 'spin'], coords),
        'eon': sim.coords_tuple(['disorder_realization', 'h', 'eigen_state'], coords),
        'E_0': sim.coords_tuple(['disorder_realization', 'h'], coords),
        'delta_E_0': sim.coords_tuple(['disorder_realization', 'h'], coords)
    },
    coords=coords
)