import os
import numpy as np
import pickle

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

from evolutionary_strategies import ES
from objective_functions import objective_function


def plot_objective(fun_name, range_min=None, range_max=None, N=50):
    objective_fun = objective_function(fun_name)['function']
    if range_min is None and range_max is None:
        range_min, range_max = objective_function(fun_name)['domain']

    x = np.linspace(range_min, range_max, N)
    y = np.linspace(range_min, range_max, N)
    X, Y = np.meshgrid(x, y)
    points = np.vstack((X.ravel(), Y.ravel())).T
    Z = objective_fun(points).reshape(N, N)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cstride=1, rstride=1, cmap='plasma')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('objective value')

    plt.show()


def plot_history(result):
    T = len(result['avg_history'])
    plt.title('Objective function')
    plt.plot(np.arange(T), result['avg_history'], label='mean')
    plt.plot(np.arange(T), result['min_history'], label='min')
    plt.plot(np.arange(T), result['max_history'], label='max')
    plt.legend()
    plt.show()


def plot_sigmas(result):
    T = len(result['avg_sigma_history'])
    plt.title('Average sigma in population')
    plt.plot(np.arange(T), result['avg_sigma_history'], label='average sigma in population')
    plt.show()


def run_test(plot_diagrams=True, **kwargs):
    function_name = kwargs['function_name']
    fun = objective_function(function_name)['function']
    domain = objective_function(function_name)['domain']
    mu = kwargs['mu']
    lam = kwargs['lam']
    d = kwargs['d']
    K = kwargs['K']
    max_iters = kwargs['max_iters']
    mode = kwargs['mode']

    es = ES(fun, d, domain)
    tau = K / np.sqrt(2*d)
    tau0 = K / np.sqrt(2*np.sqrt(d))
    res = es.optimize(mu, lam, tau, tau0, max_iters, mode)

    if plot_diagrams:
        plot_history(res)
        plot_sigmas(res)

    return res


def show_results(function_name, d, mode):
    for filename in os.listdir('results'):
        with open(f'results/{filename}', 'rb') as f:
            params = pickle.load(f)
            results = pickle.load(f)
            if params['function_name'] == function_name and \
                    params['d'] == d and params['mode'] == mode:
                plot_history(results)
                plot_sigmas(results)
                max_found = results['result'][np.newaxis, :]
                max_val = objective_function(function_name)['function'](max_found)
                print(f'Max value found: {max_val}')
                return
