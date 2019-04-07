import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

from evolutionary_strategies import ES
from objective_functions import objective_function


def plot_objective(objective_fun, range_min, range_max, N=50):
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


def run_test(**kwargs):
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

    print(f'Maximum found:{res["result"]}')
    plot_history(res)
    plot_sigmas(res)
    return res