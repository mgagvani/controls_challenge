from . import BaseController
import numpy as np
from tinyphysics import CONTROL_START_IDX

class Controller(BaseController):
    """
    A simple PID and feed-forward controller.
    PID parameters: K_p, K_i, K_d

    FF Parameters: (Given a fixed lookahead of 3)
    K_ff1, K_ff2, K_ff3
    """

    def __init__(
        self,
    ):
        # [Kp, Ki, Kd, Kff1, Kff2, Kff3, alpha_mix, integ_limit, integ_leak, deriv_alpha, u_smooth, k_roll]
        self.params = np.array([
            0.18306031, 0.09336665, -0.02,      # PID (add small damping)
            0.34426038, -0.39974145, 0.40310848, # FF (baseline)
            0.70,        # alpha_mix
            1e6,         # integ_limit (effectively disabled)
            0.0,         # integ_leak disabled
            0.5,         # derivative smoothing alpha
            0.05,        # slight control smoothing
            0.0          # k_roll disabled
        ], dtype=float)
        self.Nsteps = 3  # hard code this for now.
        self.lookahead = 20
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.prev_deriv = 0.0
        
        self.iter = 0
        self.velocities = []
        self.prev_u = 0.0

    def set_params(self, params):
        self.params = params

    def pid(self):
        return self.params[:3]

    def k_ff(self):
        return self.params[3:6]

    def alpha_mix(self):
        return float(np.clip(self.params[6], 0.0, 1.0))

    def integ_limit(self):
        return float(max(0.0, self.params[7]))

    def integ_leak(self):
        return float(np.clip(self.params[8], 0.0, 1.0))

    def deriv_alpha(self):
        return float(np.clip(self.params[9], 0.0, 1.0))

    def u_smooth(self):
        return float(np.clip(self.params[10], 0.0, 1.0))

    def k_roll(self):
        return float(self.params[11])

    def velo_factor(self, v):
        '''
        https://www.desmos.com/calculator/pepha8pun3
        At low speeds, use less of prev., high speeds use more
        '''
        return 0.4 / (1 + np.exp(-0.1 * (v - 32.5)))

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.velocities.append(state.v_ego)

        # target lataccel blend with immediate plan
        if len(future_plan.lataccel) >= 1:
            target_lataccel = self.alpha_mix() * target_lataccel + (1.0 - self.alpha_mix()) * future_plan.lataccel[0]

        # PID terms with anti-windup and derivative smoothing
        error = target_lataccel - current_lataccel
        self.error_integral = (1.0 - self.integ_leak()) * self.error_integral + error
        lim = self.integ_limit()
        if lim > 0.0:
            self.error_integral = float(np.clip(self.error_integral, -lim, lim))
        raw_deriv = error - self.prev_error
        a = self.deriv_alpha()
        self.prev_deriv = (1.0 - a) * self.prev_deriv + a * raw_deriv
        self.prev_error = error
        pid_control = (
            self.pid()[0] * error
            + self.pid()[1] * self.error_integral
            + self.pid()[2] * self.prev_deriv
        )

        # Feed-forward over smoothed future lataccel
        future_len = len(future_plan.lataccel)
        window = min(self.lookahead, future_len)
        if window <= 0:
            all_lataccel = np.zeros(self.lookahead, dtype=float)
            window = 1
        else:
            arr = np.array(future_plan.lataccel[:self.lookahead], dtype=float)
            if len(arr) < self.lookahead:
                arr = np.concatenate([arr, np.zeros(self.lookahead - len(arr))])
            all_lataccel = arr
        kernel = np.ones(window, dtype=float) / float(window)
        smooth_nextlatacc = np.convolve(all_lataccel, kernel, mode='valid')
        nextN_lataccel = np.asarray(smooth_nextlatacc[:self.Nsteps], dtype=float)
        ff_control = 0.0
        if nextN_lataccel.size > 0:
            ff_control = float(self.k_ff()[:len(nextN_lataccel)] @ nextN_lataccel)

        # roll feedforward
        roll_ff = self.k_roll() * state.roll_lataccel

        # combine
        u =  pid_control + ff_control + roll_ff
        
        # control smoothing and velocity compensation
        s = self.u_smooth()
        u = (1.0 - s) * u + s * self.prev_u
        v = state.v_ego
        vf = self.velo_factor(v)
        u = (1 - vf) * u + vf * self.prev_u
        self.prev_u = u

        return u