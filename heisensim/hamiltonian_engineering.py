import numpy as np
import qutip as qt


class WAHUHA:
    def __init__(
        self, h, taus, phase=0, min_iteration=1, max_time=None, final_relaxation=None
    ):
        self.h = h
        self.taus = taus
        self.phase = phase
        self.min_iteration = min_iteration
        self.max_time = max_time
        self.final_relaxation = final_relaxation

        if self.tpi2 > min(taus):
            raise ValueError(
                "The pi/2 pulse is larger than the time between pulses. Increase the time or the field."
            )

    @property
    def tpi2(self):
        return 2 * np.pi / (4 * self.h)

    @property
    def iterations(self):
        if self.max_time is not None:
            N = round(self.max_time / self.duration_cycle)
        else:
            N = 1
        return max(N, self.min_iteration)

    @property
    def duration_cycle(self):
        return 2 * sum(self.taus)

    @property
    def duration_total(self):
        return self.duration_cycle * self.iterations

    def get_pulse(self, h, phase):
        return (h * np.cos(phase), h * np.sin(phase), 0)

    @property
    def cycle_fields(self):
        phase = self.phase
        h = self.h
        return [
            (0, 0, 0),
            self.get_pulse(h, 0 + phase),
            (0, 0, 0),
            self.get_pulse(h, -np.pi / 2 + phase),
            (0, 0, 0),
            self.get_pulse(-h, -np.pi / 2 + phase),
            (0, 0, 0),
            self.get_pulse(-h, 0 + phase),
            (0, 0, 0),
        ]

    @property
    def cycle_durations(self):
        tau_1, tau_2, tau_3 = self.taus
        tpi2 = self.tpi2
        return [
            tau_1 - tpi2 / 2,
            tpi2,
            tau_2 - tpi2,
            tpi2,
            2 * tau_3 - tpi2,
            tpi2,
            tau_2 - tpi2,
            tpi2,
            tau_1 - tpi2 / 2,
        ]

    @property
    def fields(self):
        fields = [field for _ in range(self.iterations) for field in self.cycle_fields]
        if self.final_relaxation is not None:
            fields.append((0, 0, 0))
        return fields

    @property
    def durations(self):
        durations = self.iterations * self.cycle_durations
        if self.final_relaxation is not None:
            durations.append(self.final_relaxation)
        return durations

    def get_time_interval(self, t0, duration, dt=None):
        t1 = t0 + duration
        if dt is None:
            time_interval = [t0, t1]
        else:
            time_interval = np.arange(t0, t1, dt)
            if time_interval[-1] != t1:
                time_interval = np.append(time_interval, t1)
        return time_interval

    @staticmethod
    def gaussian(t, args):
        t0, sigma = args
        return (
            4
            / (0.9545 * np.sqrt(2 * np.pi))
            * np.exp(-((t - t0 - sigma * 2) ** 2) / (2 * sigma ** 2))
        )

    def schroedinger(
        self,
        model,
        psi0,
        dt=None,
        e_ops=None,
        options=None,
        progress_bar=None,
        _safe_mode=True,
        gaussian=False,
    ):
        if options is None:
            options = qt.Options(store_final_state=True)
        else:
            options.store_final_state = True

        H_int = model.hamiltonian_int()

        t0 = 0
        expect = [np.array([qt.expect(e_ops, psi0)])]
        times = [np.array([t0])]
        for duration, field in zip(self.durations, self.fields):
            time_interval = self.get_time_interval(t0, duration, dt)

            if gaussian:
                H_ext = [model.hamiltonian_field(*field), self.gaussian]
            else:
                H_ext = model.hamiltonian_field(*field)
            result = qt.sesolve(
                [H_int, H_ext],
                psi0,
                time_interval,
                e_ops=e_ops,
                options=options,
                progress_bar=progress_bar,
                _safe_mode=_safe_mode,
                args=[t0, duration / 4],
            )

            t0 = t0 + duration
            psi0 = result.final_state
            expect.append(np.array(result.expect)[:, 1:])
            times.append(time_interval[1:])
        return np.concatenate(times), np.concatenate(expect, axis=1)


class WAHUHA_amp_errors(WAHUHA):
    def __init__(
        self,
        h,
        taus,
        phase=0,
        min_iteration=1,
        max_time=None,
        phase_error=0,
        final_relaxation=None,
    ):
        super().__init__(
            h,
            taus,
            phase=phase,
            min_iteration=min_iteration,
            max_time=max_time,
            final_relaxation=final_relaxation,
        )
        self.phase_error = phase_error
        if phase != 0:
            raise ValueError(
                "phase needs to be zero. Otherwise, use standard WAHUHA sequence."
            )

    def get_pulse(self, h, phase):
        n1, n2 = list(self.phase_error * np.random.randn(2))
        if np.isclose(phase, 0):
            return (h * np.sqrt(1 - n1 ** 2 - n2 ** 2), h * n1, h * n2)
        else:psett
            return (h * n2, h * np.sqrt(1 - n1 ** 2 - n2 ** 2), h * n1)


def schroedinger_sequence(
    model,
    psi0,
    durations,
    fields,
    dt=None,
    e_ops=None,
    args=None,
    options=None,
    progress_bar=None,
    _safe_mode=True,
):
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
        result = qt.sesolve(
            H_int + H_ext,
            psi0,
            time_interval,
            e_ops=e_ops,
            args=args,
            options=options,
            progress_bar=progress_bar,
            _safe_mode=_safe_mode,
        )

        psi0 = result.final_state
        expect.append(np.array(result.expect)[:, 1:])
        times.append(time_interval[1:])
    return np.concatenate(times), np.concatenate(expect, axis=1)
