from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from concurrent.futures import ProcessPoolExecutor
from random import sample
from functools import partial
import numpy as np
import argparse
import datetime
import json
import os

from tinyphysics import CONTROL_START_IDX, run_rollout_controller
from controllers.pid_ff import Controller as PIDFFController

import cma

class ControlEvolver:
    def __init__(self, controller, data_path, model_path, n_rollouts=100, n_segs=100):
        self.controller = controller
        self.n_rollouts = n_rollouts
        self.data_path = data_path
        self.n_segs = n_segs
        self.model_path = model_path

        self.load_data(data_path)

    def load_data(self, data_path):
        """
        Load data as shown in eval.py
        """
        data_path = Path(self.data_path)
        assert data_path.is_dir(), "data_path should be a directory"

        self.files = sorted(data_path.iterdir()) # [:self.n_segs] (sample this much each time)


    def fitness_function(self, controller, params, verbose=False):
        """
        Evaluate the controller's performance (accel/jerk cost)
        """
        # for now, only evolve FF params
        controller.set_params(params)

        rollout_partial = partial(run_rollout_controller, controller=controller, model_path=self.model_path, debug=False)

        # sample the whole dataset
        files = sample(self.files, self.n_rollouts)

        # cheese the eval.py
        # files = sample(self.files[:5000], self.n_rollouts)

        # for d, data_file in enumerate(tqdm(files, total=self.n_segs)):
        #     cost, target_lataccel, current_lataccel = rollout_partial(data_file)
        #     print(f"Rollout {d}: {cost}")
        if verbose:
            results = process_map(rollout_partial, files, max_workers=16, chunksize=5)
        else:
            results = ProcessPoolExecutor(max_workers=16, max_tasks_per_child=10).map(rollout_partial, files)
        rollout_results = [result[0] for result in results]
        # each rollout result is {'lataccel_cost': cost, 'jerk_cost': jerk_cost, 'total_cost': cost}
        total_costs = [result['total_cost'] for result in rollout_results]
        return np.mean(total_costs)


    def evolve_pidff_controller(self, initial_params=None, sigma=0.3, max_iter=150, popsize=30, bounds=None, stop_below=None, checkpoint_path=None, resume_from=None):
        """
        Evolve PID+FF controller using the CMA-ES evolution strategy
        """
        if resume_from is not None:
            initial_params = np.load(resume_from)
        elif initial_params is None:
            # derive dimensionality from controller
            initial_params = np.array(self.controller.params, dtype=float)

        # plusminus_bounds = np.array([0.025, 0.025, 0.025, 1, 1, 1])
        # bounds = [[i - b, i + b] for i, b in zip(initial_params, plusminus_bounds)]
        
        def fitness(params):
            return self.fitness_function(self.controller, params)
        
        if bounds is None:
            # per-parameter bounds: PID/FF near [-2.5, 2.5], alpha/ratios in [0,1], integrator clamp [0, 50]
            lower = np.array([-2.5, -2.5, -2.5,  -2.5, -2.5, -2.5,   0.0,   0.0,  0.0,  0.0,  0.0, -2.5], dtype=float)
            upper = np.array([ 2.5,  2.5,  2.5,   2.5,  2.5,  2.5,   1.0,  50.0,  1.0,  1.0,  1.0,  2.5], dtype=float)
            bounds = (lower, upper)

        es = cma.CMAEvolutionStrategy(x0=initial_params, 
                                      sigma0=sigma,
                                      options=
                                      {'tolstagnation': 0,
                                       'bounds': bounds,
                                       'popsize': popsize,
                                       'maxiter': max_iter,},
                                      )
        
        best_params = initial_params
        best_fitness = fitness(initial_params)
        print(f"Initial fitness: {best_fitness}")

        # file for logging
        os.makedirs('tmp', exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = f"tmp/cmaes_log_{timestamp}.txt"
        log_file = open(log_path, "w")
        ckpt_path = checkpoint_path or "tmp/best_pidff_params.npy"
        meta_path = "tmp/best_pidff_meta.json"
        
        iteration = 0
        history = []
        try:
            with tqdm(total=max_iter, desc="Evolving controller") as pbar:
                while not es.stop() and iteration < max_iter:
                    solutions = es.ask()
                    fitnesses = []
                    for s in solutions:
                        f = fitness(s)
                        fitnesses.append(f)
                        print(f"Solution: {s}, Fitness: {f}")
                    # fitnesses = [fitness(s) for s in solutions]
                    es.tell(solutions, fitnesses)
                    
                    current_best_idx = np.argmin(fitnesses)
                    current_best_fitness = fitnesses[current_best_idx]
                    current_best_solution = solutions[current_best_idx]
                    
                    if current_best_fitness < best_fitness:
                        best_fitness = current_best_fitness
                        best_params = current_best_solution
                        # checkpoint best
                        np.save(ckpt_path, best_params)
                        with open(meta_path, 'w') as mf:
                            json.dump({
                                'best_fitness': float(best_fitness),
                                'best_params': [float(x) for x in best_params],
                                'iteration': int(iteration),
                                'log_path': log_path,
                                'timestamp': timestamp,
                            }, mf)
                    
                    history.append((iteration, best_fitness))
                    now_str = datetime.datetime.utcnow().isoformat()
                    print(f"Iteration {iteration}, Best fitness: {best_fitness}, time: {now_str}")
                    log_file.write(f"Iteration {iteration}, Best fitness: {best_fitness}, time: {now_str}, Best params: {best_params}\n")
                    # append progress CSV
                    progress_csv = 'tmp/cmaes_progress.csv'
                    write_header = not os.path.exists(progress_csv)
                    with open(progress_csv, 'a') as cf:
                        if write_header:
                            cf.write('iteration,timestamp,best_fitness\n')
                        cf.write(f"{iteration},{now_str},{best_fitness}\n")
                    log_file.flush()
                    # early stop if threshold achieved
                    if stop_below is not None and best_fitness < stop_below:
                        break
                    
                    iteration += 1
                    pbar.update(1)
                    pbar.set_postfix({'fitness': best_fitness})
        
        except KeyboardInterrupt:
            print("keyboard interrupt...")

        log_file.close()
        
        # Set the controller to the best parameters found
        self.controller.params = best_params
        
        # Plot convergence
        if history:
            iterations, fitnesses = zip(*history)
            plt.figure(figsize=(10, 6))
            plt.plot(iterations, fitnesses)
            plt.title('CMA-ES Convergence')
            plt.xlabel('Iteration')
            plt.ylabel('Fitness (Cost)')
            plt.grid(True)
            plt.savefig('cmaes_convergence.png')
            plt.close()
        
        return best_params, best_fitness

    

if __name__ == "__main__":
    '''
    python evolve_controller.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 125
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_rollouts", type=int, default=100)
    parser.add_argument("--num_segs", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--max_iter", type=int, default=150)
    parser.add_argument("--popsize", type=int, default=30)
    parser.add_argument("--stop_below", type=float, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    args = parser.parse_args()

    controller = PIDFFController()
    
    evolver = ControlEvolver(
        controller=controller,
        data_path=args.data_path,
        model_path=args.model_path,
        n_rollouts=args.num_rollouts,
        n_segs=args.num_segs
    )

    # Run evolution to find optimal parameters
    best_params, best_fitness = evolver.evolve_pidff_controller(
        sigma=args.sigma,
        max_iter=args.max_iter,
        popsize=args.popsize,
        stop_below=args.stop_below,
        checkpoint_path=args.checkpoint_path,
        resume_from=args.resume_from,
    )
    print(f"Best params: {best_params}, Best fitness: {best_fitness}")
    
    # Just test with default parameters
    # cost = evolver.fitness_function(controller, np.array([0.25, 0.125, 0.0625]), True) # just as a starter
    # print(f"Cost: {cost}")
