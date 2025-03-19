from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  A simple PID and feed-forward controller.
  PID parameters: K_p, K_i, K_d

  FF Parameters: (Given a fixed lookahead of 3)
  K_ff1, K_ff2, K_ff3
  """
  def __init__(self,):
    self.p = 0.3
    self.i = 0.05
    self.d = -0.1
    self.k_ff = np.array([0.03481429, 0.20785773, 0.13804611]) # just as a starter
    self.Nsteps = len(self.k_ff)
    self.error_integral = 0
    self.prev_error = 0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
        # PID
        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        pid_control = self.p * error + self.i * self.error_integral + self.d * error_diff

        # FF
        nextN_lataccel = np.array(future_plan.lataccel[:self.Nsteps])            
        ff_control = self.k_ff[:len(nextN_lataccel)] @ nextN_lataccel; # dot prod, ensure dimension match   

        return pid_control + ff_control
