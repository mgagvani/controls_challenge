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
        [0.12933709,  0.10433873, -0.00234403, -0.00918009,  0.20953831,  0.19877911]
        )  # [PID], [FF] (each 3)
        self.Nsteps = 3  # hard code this for now.
        self.lookahead = 20
        self.error_integral = 0
        self.prev_error = 0
        
        self.iter = 0
        self.velocities = []

    def set_params(self, params):
        self.params = params

    def pid(self):
        return self.params[:3]

    def k_ff(self):
        return self.params[3:]

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.velocities.append(state.v_ego)

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
        nextN_lataccel = np.array(future_plan.lataccel[:self.Nsteps])
        ff_control = self.k_ff()[:len(nextN_lataccel)] @ nextN_lataccel # dot prod, ensure dimension match

        return pid_control + ff_control