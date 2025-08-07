import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel, STEER_RANGE, CONTROL_START_IDX, COST_END_IDX
from controllers import BaseController

class RLController(BaseController):
    def __init__(self, action):
        self.action = action

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        return self.action

class TinyPhysicsEnv(gym.Env):
    def __init__(self, model_path, data_path):
        super(TinyPhysicsEnv, self).__init__()
        self.model_path = model_path
        self.data_path = data_path

        self.sim_model = TinyPhysicsModel(self.model_path, debug=False)
        self.sim = TinyPhysicsSimulator(self.sim_model, self.data_path, controller=None)

        # Action space: steer command
        self.action_space = gym.spaces.Box(low=STEER_RANGE[0], high=STEER_RANGE[1], shape=(1,), dtype=np.float32)

        # Observation space: [roll_lataccel, v_ego, a_ego, target_lataccel, current_lataccel]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        return self._get_obs(), {}

    def step(self, action):
        action = action[0]
        self.sim.controller = RLController(action)

        # We need to run the simulator until the next control step
        self.sim.step()

        obs = self._get_obs()
        cost = self.sim.compute_cost()['total_cost']

        if np.isnan(cost) or np.isinf(cost):
            reward = -1000
        else:
            reward = -cost

        reward = np.clip(reward, -1000, 1000)

        done = self.sim.step_idx >= len(self.sim.data) - 1

        return obs, reward, done, False, {}

    def _get_obs(self):
        state = self.sim.state_history[-1]
        target_lataccel = self.sim.target_lataccel_history[-1]
        current_lataccel = self.sim.current_lataccel_history[-1]
        return np.array([state.roll_lataccel, state.v_ego, state.a_ego, target_lataccel, current_lataccel], dtype=np.float32)

    def render(self, mode='human'):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    from tinyphysics import download_dataset, DATASET_PATH
    import os

    print(f"Dataset path: {DATASET_PATH}")
    if not DATASET_PATH.exists():
        print("Dataset not found. Downloading...")
        download_dataset()
        print("Download complete.")
    else:
        print("Dataset found.")

    print(f"Files in dataset path: {os.listdir(DATASET_PATH)}")

    # Path to the model and data
    MODEL_PATH = "models/tinyphysics.onnx"

    # use the first segment for training
    files = sorted(os.listdir(DATASET_PATH))
    DATA_PATH = os.path.join(DATASET_PATH, files[0])

    # Create the Gymnasium environment
    env = TinyPhysicsEnv(model_path=MODEL_PATH, data_path=DATA_PATH)

    # Instantiate the PPO agent
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=20000)

    # Save the trained model
    model.save("rl_controller")

    print("Training complete. Model saved to rl_controller.zip")
