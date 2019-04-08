import numpy as np
from time import time

class ES(object):

    def __init__(self, objective_function, dimensions, domain):
        self.objective_function = objective_function
        self.dimensions = dimensions
        self.domain = domain
        self.MAX_SIGMA = 100000.

    def optimize(self, mu, lam, tau, tau0, max_iterations, mode='plus'):
        total_start_timestamp = time()
        xs_current, sigmas_current = self.random_population(mu)
        objective_values = self.evaluate_population(xs_current)

        max_history = np.empty(max_iterations, dtype=np.float64)
        min_history = np.empty(max_iterations, dtype=np.float64)
        avg_history = np.empty(max_iterations, dtype=np.float64)
        avg_sigma_history = np.empty(max_iterations, dtype=np.float64)

        for t in range(max_iterations):
            iter_start_timestamp = time()

            max_history[t] = np.max(objective_values)
            min_history[t] = np.min(objective_values)
            avg_history[t] = np.mean(objective_values)
            avg_sigma_history[t] = np.mean(sigmas_current)

            xs_children, sigmas_children = self.parent_selection(xs_current, sigmas_current, objective_values, lam)
            xs_mutated, sigmas_mutated = self.mutation(xs_children, sigmas_children, tau, tau0)
            xs_current, sigmas_current, objective_values = self.replacement_and_evaluation(
                xs_current, sigmas_current, xs_mutated, sigmas_mutated, mode)

            iter_finish_timestamp = time()
            iter_time_elapsed = iter_finish_timestamp - iter_start_timestamp
            eta = iter_time_elapsed * (max_iterations - t)
            if t % 200 == 0:
                print(f't = {t}, max = {max_history[t]}, avg_sigma = {avg_sigma_history[t]:.2f}, iter time = {iter_time_elapsed:.2f}s ETA = {eta:.2f}s')

        total_finish_timestamp = time()
        total_time_elapsed = total_finish_timestamp - total_start_timestamp

        return {'result': xs_current[objective_values.argmax()],
                'max_history': max_history,
                'min_history': min_history,
                'avg_history': avg_history,
                'avg_sigma_history': avg_sigma_history,
                'total_time_elapsed': total_time_elapsed}

    def random_population(self, mu):
        return np.random.uniform(self.domain[0], self.domain[1], (mu, self.dimensions)), np.random.randn(mu, self.dimensions)

    def evaluate_population(self, population):
        return self.objective_function(population)

    def parent_selection(self, xs, sigmas, objective_values, lam):
        population_size = objective_values.shape[0]
        fitness_values = objective_values.max() - objective_values
        if fitness_values.sum() > 0:
            fitness_values = fitness_values / fitness_values.sum()
        else:
            fitness_values = np.ones(population_size) / population_size
        parent_indices = np.random.choice(population_size, lam, True, fitness_values).astype(np.int64)
        return xs[parent_indices], sigmas[parent_indices]

    def mutation(self, xs, sigmas, tau, tau0):
        epsilon_0 = np.random.normal(0, tau0)
        epsilon_i_for_sigmas = np.random.normal(np.zeros_like(sigmas), sigmas ** 2)

        new_sigmas = np.clip(sigmas * np.exp(epsilon_0 + epsilon_i_for_sigmas), -self.MAX_SIGMA, self.MAX_SIGMA)
        epsilon_i_for_xs = np.random.normal(np.zeros_like(new_sigmas), new_sigmas ** 2)
        new_xs = np.clip(xs + epsilon_i_for_xs, self.domain[0], self.domain[1])
        return new_xs, new_sigmas

    def replacement_and_evaluation(self, xs, sigmas, xs_children, sigmas_children, mode):
        if mode == 'plus':
            xs_all = np.vstack((xs, xs_children))
            sigmas_all = np.vstack((sigmas, sigmas_children))
        elif mode == 'comma':
            xs_all = xs_children
            sigmas_all = sigmas_children
        else:
            raise ValueError()

        objective_values = self.evaluate_population(xs_all)
        population_indices = objective_values.argsort()[-xs.shape[0]:]
        return xs_all[population_indices], sigmas_all[population_indices], objective_values[population_indices]
