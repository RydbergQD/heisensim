import numpy as np
import qutip as qt


def wahuha(taus, h, N=1, max_time=None, phase=0):
    if max_time is not None:
        N = round(max_time / (2 * sum(taus)))
    tau_1, tau_2, tau_3 = taus
    tpi2 = 2*np.pi/(4*h)
    if tpi2 > min(taus):
        raise ValueError("The pi/2 pulse is larger than the time between pulses. Increase the time or the field.")
    durations = [
        tau_1 - tpi2/2, tpi2,
        tau_2 - tpi2, tpi2,
        2*tau_3 - tpi2, tpi2,
        tau_2 - tpi2, tpi2,
        tau_1 - tpi2/2
    ]
    fields = [
        (0, 0, 0), (h*np.cos(phase), np.sin(phase), 0),
        (0, 0, 0), (np.sin(phase), -h*np.cos(phase), 0),
        (0, 0, 0), (np.sin(phase), h*np.cos(phase), 0),
        (0, 0, 0), (-h*np.cos(phase), np.sin(phase), 0),
        (0, 0, 0)
    ]
    return durations * N, fields * N

def get_times_sequence(durations, t0=0):
    times_sequence = [t0]
    for duration in durations:
        t0 = t0 + duration
        times_sequence.append(t0)
    return times_sequence    

def get_t_snippet(t_list, t0, t1):
    t_list = np.array(t_list)
    t_snippet = t_list[(t0 <= t_list) & (t1 >= t_list)]
    return t_snippet
    
def schroedinger_sequence(model, psi0, durations, fields, dt=None, e_ops=None, args=None, options=None, progress_bar=None, _safe_mode=True):
    if options is None:
        options = qt.Options(store_final_state=True)
    else:
        options.store_final_state = True

    H_int = model.hamiltonian_int()
    
    t0 = 0
    expect = [np.array([qt.expect(e_ops, psi0)])]
    times = [np.array([t0])]
    for duration, field in zip(durations, fields):
        t1 = t0 + duration
        if dt is None:
            time_interval = [t0, t1]
        else:
            time_interval = np.arange(t0, t1, dt)
            if time_interval[-1] != t1:
                time_interval = np.append(time_interval, t1)
        t0 = t1

        H_ext = model.hamiltonian_field(*field)
        result = qt.sesolve(H_int + H_ext, psi0, time_interval, e_ops=e_ops, args=args, options=options, progress_bar=progress_bar, _safe_mode=_safe_mode)
        
        psi0 = result.final_state
        expect.append(np.array(result.expect)[:, 1:]) 
        times.append(time_interval[1:])
    return np.concatenate(times), np.concatenate(expect, axis=1)
