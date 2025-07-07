import pandas as pd
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

from controllers import BaseController

from tinyphysics import (
    TinyPhysicsModel, TinyPhysicsSimulator, State, FuturePlan, 
    STEER_RANGE, LAT_ACCEL_COST_MULTIPLIER, CONTROL_START_IDX, CONTEXT_LENGTH, DEL_T, run_rollout
)
from controllers.pid_ff import Controller as PIDFFController

'''
The idea here is to use SLSQP to create optimal control values for each rollout in the dataset.
Basically, use scipy.optimize.minimize, given the cost function, initial guess (from PIDFF) and bounds from tinyphysics.
'''
'''
Another approach is receding horizon control (MPC). Over a short time horizon, we can find an optimal control action 
that minimizes the cost function over the next few seconds. Using the cross-entropy method, we can iteratively
refine the control action based on the costs.
'''

HORIZON = 20 # timesteps

def compute_costs_for_actions(sim: TinyPhysicsSimulator, actions: np.ndarray):
    """
    Given a simulation, compute the cost for each action sequence in actions [n_samples, seq_len].
    Basically a rehash of control_step and compute_costs. 
    However, we want to spin up many TinyPhysicsModels, each with the history/info of `sim` and run each action sequence through the model.
    Note that the `sim` object doesn't actually see the actions. We will propagate forward near-ideal PID actions in order to get to the 
    next timestep, since we don't want a bad action sequence to permanently affect future predictions based on a `sim`.
    """
    n_samples, seq_len = actions.shape
    costs = np.zeros((n_samples, 3))  # lataccel_cost, jerk_cost, total_cost

    # create payload for TinyPhysicsModel from sim: (sim_states: List[State], actions: List[float], past_preds: List[float])
    for i, action_seq in enumerate(actions): # for each guessed action sequence, 
        sim_states = sim.state_history[-CONTEXT_LENGTH:].copy()
        actions = sim.action_history[-CONTEXT_LENGTH:].copy()
        past_preds = sim.current_lataccel_history[-CONTEXT_LENGTH:].copy() 
        physics_model = TinyPhysicsModel("models/tinyphysics.onnx", debug=False) # instantiate a fresh physics model
        target_lataccel = sim.data["target_lataccel"].values[sim.step_idx:sim.step_idx + seq_len] # get the target lataccel for the next `seq_len` timesteps
        pred_lataccel_seq = [] # hold the predicted lataccel from tinyphysics
        for timestep in range(seq_len): # roll out the small action sequence iteratively
            actions.append(action_seq[timestep]) # this is basically the control_step
            if len(actions) > CONTEXT_LENGTH:
                actions = actions[-CONTEXT_LENGTH:]  # keep only the last CONTEXT_LENGTH actions (check this?)
            pred_lataccel = physics_model.get_current_lataccel(sim_states, actions, past_preds)
            pred_lataccel_seq.append(pred_lataccel) 
            past_preds.append(pred_lataccel) # the last line of sim_step
            if len(past_preds) > CONTEXT_LENGTH: # ensure we are always <= CONTEXT_LENGTH
                past_preds = past_preds[-CONTEXT_LENGTH:]
            # NOTE: I'm not sure if we want to update according to GT or what is predicted here.
            next_idx = sim.step_idx + timestep # the last bit of this inner loop is incrementing the sim according to what happened in ground truth
            if next_idx < len(sim.data):
                next_state, _, _ = sim.get_state_target_futureplan(next_idx) 
                sim_states.append(next_state)
                if len(sim_states) > CONTEXT_LENGTH: # ensure we are always <= CONTEXT
                    sim_states = sim_states[-CONTEXT_LENGTH:]  # keep only the last CONTEXT_LENGTH states
        # now that we've rolled out the for our given action sequence, compute costs over most recent seq_len steps
        pred_lataccel = np.array(pred_lataccel_seq)
        lat_accel_cost = np.mean((target_lataccel - pred_lataccel)**2) * 100
        jerk_cost = np.mean((np.diff(pred_lataccel) / DEL_T)**2) * 100
        total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
        costs[i] = np.array([lat_accel_cost, jerk_cost, total_cost])
        print(f"Action seq {i+1}/{n_samples} costs: [{costs[i][0]:.2f}, {costs[i][1]:.2f}, {costs[i][2]:.2f}]", end='\r')

    return costs



def cem_step(sim: TinyPhysicsSimulator, n_samples=200, n_elite=10, seq_len=HORIZON):
    """
    Cross-Entropy Method to find optimal control action at `state` for the next `seq_len` timesteps.

    Args:
        state (State): Current vehicle state we use as a starting point.
        n_samples (int): Number of random samples to try as control actions.
        n_elite (int): Number of elite samples to use as a basis for sampling again.
        seq_len (int): Number of timesteps to optimize over.
    """

    mu, sigma = np.zeros(seq_len), np.ones(seq_len) * 0.5 # not a single action, but a sequence of actions
    for _ in range(20):
        actions = np.random.randn(n_samples, seq_len) * sigma + mu # sample actions over time from a normal distribution
        costs = compute_costs_for_actions(sim, actions) 
        elite_indices = np.argsort(costs[:, 2])[:n_elite]  # get indices of the n_elite best actions
        mu, sigma = np.mean(actions[elite_indices], axis=0), np.std(actions[elite_indices], axis=0)+1e-3 
        print(f"Best cost @ {_}: {costs[elite_indices[0], 2]:.2f} mean sigma = {np.mean(sigma):.4f}")
    
    return np.clip(mu, *STEER_RANGE) # now, mu should have converged upon a good action sequence

def cem_rollout(sim: TinyPhysicsSimulator, n_samples=200, n_elite=10, seq_len=HORIZON):
    """
    Run a rollout using the CEM method to find a near-optimal control action sequence.
    Procedure:
    1. Load the CSV data into `sim`. 
    2. At timestep t, use CEM to compute a short horizon of optimal control actions. Apply only the first action to t+1. 
    3. Repeat until the end of the dataset.
    """

    for _ in range(CONTEXT_LENGTH, len(sim.data)):
        # unroll out sim.step here, since we are changing some bits
        # pre control-step
        state, target, futureplan = sim.get_state_target_futureplan(sim.step_idx)
        sim.state_history.append(state)
        sim.target_lataccel_history.append(target)
        sim.futureplan = futureplan
        # control step
        actions = cem_step(sim, n_samples=n_samples, n_elite=n_elite, seq_len=seq_len)
        action = actions[0]
        if sim.step_idx < CONTROL_START_IDX:
            action = sim.data['steer_command'].values[sim.step_idx]  
        action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
        sim.action_history.append(action) 
        # post control-step
        sim.sim_step(sim.step_idx)
        sim.step_idx += 1

    # plot and return at the end
    fig, ax = plt.subplots(4, figsize=(12, 14), constrained_layout=True)
    sim.plot_data(ax[0], [(sim.target_lataccel_history, 'Target lataccel'), (sim.current_lataccel_history, 'Current lataccel')], ['Step', 'Lateral Acceleration'], 'Lateral Acceleration')
    sim.plot_data(ax[1], [(sim.action_history, 'Action')], ['Step', 'Action'], 'Action')
    sim.plot_data(ax[2], [(np.array(sim.state_history)[:, 0], 'Roll Lateral Acceleration')], ['Step', 'Lateral Accel due to Road Roll'], 'Lateral Accel due to Road Roll')
    sim.plot_data(ax[3], [(np.array(sim.state_history)[:, 1], 'v_ego')], ['Step', 'v_ego'], 'v_ego')

    plt.savefig(f"tmp/cem_rollout_{Path(sim.data_path).stem}.png")

    return sim.compute_cost()


if __name__ == "__main__":
    # we want to compare and plot the results of a single rollout optimized with CEM vs. simply using PIDFF.

    model_path = "models/tinyphysics.onnx"
    data_path = "data/00000.csv"

    cem_sim = TinyPhysicsSimulator(TinyPhysicsModel(model_path, debug=False), data_path, controller=...)
    pidff_sim = TinyPhysicsSimulator(TinyPhysicsModel(model_path, debug=False), data_path, controller=PIDFFController(), debug=True)

    # pidff_sim.rollout()
    cem_rollout(cem_sim)






    