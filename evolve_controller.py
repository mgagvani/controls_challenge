from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from concurrent.futures import ProcessPoolExecutor
from random import sample
from functools import partial
import numpy as np
import argparse

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
        controller.k_ff = params 

        rollout_partial = partial(run_rollout_controller, controller=controller, model_path=self.model_path, debug=False)

        # randomly sample some data
        costs = []
        sample_rollouts = []

        # sample the whole dataset
        files = sample(self.files, self.n_rollouts)

        # cheese the eval.py
        # files = sample(self.files[:5000], self.n_rollouts)

        # for d, data_file in enumerate(tqdm(files, total=self.n_segs)):
        #     cost, target_lataccel, current_lataccel = rollout_partial(data_file)
        #     print(f"Rollout {d}: {cost}")
        if verbose:
            results = process_map(rollout_partial, files, max_workers=48, chunksize=5)
        else:
            results = ProcessPoolExecutor(max_workers=48, max_tasks_per_child=5).map(rollout_partial, files)
        rollout_results = [result[0] for result in results]
        # each rollout result is {'lataccel_cost': cost, 'jerk_cost': jerk_cost, 'total_cost': cost}
        total_costs = [result['total_cost'] for result in rollout_results]
        return np.mean(total_costs)


    def evolve_pidff_controller(self, initial_params=None, sigma=0.3, max_iter=50):
        """
        Evolve PID+FF controller using the CMA-ES evolution strategy
        """
        if initial_params is None:
            initial_params = np.array([0.25, 0.125, 0.0625])
        
        def fitness(params):
            return self.fitness_function(self.controller, params)
        
        es = cma.CMAEvolutionStrategy(x0=initial_params, 
                                      sigma0=sigma,
                                      options=
                                      {'tolstagnation': 0,
                                       'bounds': [0, 10],
                                       'popsize': 25,},
                                      )
        
        best_params = initial_params
        best_fitness = fitness(initial_params)
        print(f"Initial fitness: {best_fitness}")
        
        iteration = 0
        history = []
        try:
            with tqdm(total=max_iter, desc="Evolving controller") as pbar:
                while not es.stop() and iteration < max_iter:
                    solutions = es.ask()
                    fitnesses = []
                    for s in solutions:
                        fitness = fitness(s)
                        fitnesses.append(fitness)
                        print(f"Solution: {s}, Fitness: {fitness}")
                    # fitnesses = [fitness(s) for s in solutions]
                    es.tell(solutions, fitnesses)
                    
                    current_best_idx = np.argmin(fitnesses)
                    current_best_fitness = fitnesses[current_best_idx]
                    current_best_solution = solutions[current_best_idx]
                    
                    if current_best_fitness < best_fitness:
                        best_fitness = current_best_fitness
                        best_params = current_best_solution
                    
                    history.append((iteration, best_fitness))
                    print(f"Iteration {iteration}, Best fitness: {best_fitness}, Best params: {best_params}")
                    
                    iteration += 1
                    pbar.update(1)
                    pbar.set_postfix({'fitness': best_fitness})
        
        except KeyboardInterrupt:
            print("keyboard interrupt...")
        
        # Set the controller to the best parameters found
        self.controller.k_ff = best_params
        
        # Plot convergence
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
    best_params, best_fitness = evolver.evolve_pidff_controller()
    print(f"Best params: {best_params}, Best fitness: {best_fitness}")
    
    # Just test with default parameters
    # cost = evolver.fitness_function(controller, np.array([0.25, 0.125, 0.0625]), True) # just as a starter
    # print(f"Cost: {cost}")
