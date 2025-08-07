import numpy as np
from stable_baselines3 import PPO

from controllers import BaseController

class Controller(BaseController):
    def __init__(self):
        self.model = PPO.load("rl_controller")

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        obs = np.array([state.roll_lataccel, state.v_ego, state.a_ego, target_lataccel, current_lataccel], dtype=np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        return action[0]
