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
        self.params = self.params = np.array(
        # [ 0.19971562,  0.09430931, -0.05626948, -0.17563558,  0.47475137, -0.02800372]
        # [ 0.0893906,   0.11858476,  0.01039364,  0.18927849,  0.21582582, -0.00069932]
        # [ 0.12764586,  0.09719727, -0.004143,   -0.01876578,  0.50789625, -0.14295344]
        # [ 0.16619618,  0.09902905,  0.03087365, -0.01971456 , 0.5015729, -0.10662882]
        # [0.16213236,  0.10320203,  0.04349641, -0.0215636,   0.48318652, -0.11382189]
        # [ 0.16341881,  0.10287497,  0.04711446, -0.02289423,  0.4843983,  -0.11057394]
        [ 0.18306031,  0.09336665,  0.00482326,  0.34426038, -0.39974145,  0.40310848]
        )  # [PID], [FF] (each 3)
        self.Nsteps = 3  # hard code this for now.
        self.lookahead = 20
        self.error_integral = 0
        self.prev_error = 0
        
        self.iter = 0
        self.velocities = []
        self.prev_u = 0

    def set_params(self, params):
        self.params = params

    def pid(self):
        return self.params[:3]

    def k_ff(self):
        return self.params[3:]

    def velo_factor(self, v):
        '''
        https://www.desmos.com/calculator/pepha8pun3
        At low speeds, use less of prev., high speeds use more
        '''
        return 0.4 / (1 + np.exp(-0.1 * (v - 32.5)))

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.velocities.append(state.v_ego)

        # target lataccel. current * k + (1-k) * future
        if len(future_plan.lataccel) >= 2:
            target_lataccel = target_lataccel * 0.7 + future_plan.lataccel[0] * 0.3

        # PID
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        pid_control = (
            self.pid()[0] * error
            + self.pid()[1] * self.error_integral
            + self.pid()[2] * error_diff
        )

        # FF
        # conv with ones over future 2 sec to smooth feedforward
        lookahead = min(self.lookahead, len(future_plan))
        all_lataccel = future_plan.lataccel
        if lookahead == 0:
            all_lataccel = np.zeros(self.lookahead)
        elif lookahead < self.lookahead:
            all_lataccel = np.concatenate((all_lataccel, np.zeros(self.lookahead - lookahead)))
        else:
            all_lataccel = all_lataccel[:self.lookahead]
        smooth_nextlatacc = np.convolve(all_lataccel, np.ones(lookahead)/lookahead, mode='valid')
        nextN_lataccel = np.array(smooth_nextlatacc[:self.Nsteps])
        ff_control = self.k_ff()[:len(nextN_lataccel)] @ nextN_lataccel # dot prod, ensure dimension match

        u =  pid_control + ff_control
        
        # velocity compensation
        v = state.v_ego
        vf = self.velo_factor(v)
        u = (1 - vf) * u + vf * self.prev_u
        self.prev_u = u

        return u