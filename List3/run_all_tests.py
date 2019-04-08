import numpy as np
import pickle
from utils import run_test, plot_objective, plot_history, plot_sigmas

if __name__ == '__main__':
    ds = [10, 30, 50, 80, 150]
    funs = ['rastrigin', 'schwefel', 'ackley', 'rosenbrock']

    for function_name in funs:
        for d in ds:
            print('=================================================')
            print(f'Processing: d = {d}, function = {function_name}')
            params = {'function_name': function_name,
                      'mu': 100000,
                      'lam': 200000,
                      'd': d,
                      'K': 1,
                      'max_iters': 1000,
                      'mode': 'comma'}

            res = run_test(plot_diagrams=False, **params)
            filename = hash(frozenset(params.items()))
            with open(f'results/res{filename}', 'wb+') as f:
                print(f'Saving results to: results/res{filename}')
                pickle.dump(params, f)
                pickle.dump(res, f)
