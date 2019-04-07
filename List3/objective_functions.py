import numpy as np

def negative(f):
    return lambda x: -f(x)

def ackley(x, a=20., b=0.2, c=2*np.pi):
    d = x.shape[1]
    return -a * np.exp(-(b / d) * np.sum(x ** 2, axis=1)) - np.exp(np.sum(np.cos(c*x), axis=1) / d) + a + np.exp(1)

def griewank(x):
    d = x.shape[1]
    return 1 + np.sum(x ** 2, axis=1) / 4000. - np.prod(np.cos(x / np.arange(1, d+1)), axis=1)

def rastrigin(x, A=10.):
    d = x.shape[1]
    return A*d + np.sum(x ** 2 - A * np.cos(2*np.pi*x), axis=1)

def schwefel(x):
    d = x.shape[1]
    return 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)

def rosenbrock(x):
    return np.sum(100 * ((x[:, ::2] ** 2 - x[:, 1::2]) ** 2) + (x[:, ::2] - 1) ** 2, axis=1)

def objective_function(name):
    functions = {'ackley': negative(ackley),
                 'griewank': negative(griewank),
                 'rastrigin': negative(rastrigin),
                 'schwefel': negative(schwefel),
                 'rosenbrock': negative(rosenbrock)}

    domains = {'ackley': (-5., 5.),
               'griewank': (-600., 600.),
               'rastrigin': (-5.12, 5.12),
               'schwefel': (-500., 500.),
               'rosenbrock': (-5, 10.)}

    return {'function': functions[name], 'domain': domains[name] }
