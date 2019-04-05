import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d


def plot_objective(objective_function, range_min, range_max, N=50):
    x = np.linspace(range_min, range_max, N)
    y = np.linspace(range_min, range_max, N)
    X, Y = np.meshgrid(x, y)
    points = np.vstack((X.ravel(), Y.ravel())).T
    Z = objective_function(points).reshape(N, N)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    print('halo')
    ax.plot_surface(X, Y, Z, cstride=1, rstride=1, cmap='plasma')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('objective value')

    plt.show()

def plot_history(result):
    T = len(result['avg_history'])
    plt.plot(np.arange(T), result['avg_history'], label='mean')
    plt.plot(np.arange(T), result['min_history'], label='min')
    plt.plot(np.arange(T), result['max_history'], label='max')
    plt.legend()
    plt.show()
