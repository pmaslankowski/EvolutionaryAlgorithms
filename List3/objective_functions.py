import numpy as np

def ackley(x, a=20., b=0.2, c=2*np.pi):
    d = x.shape[1]
    return -(-a * np.exp(-(b / d) * np.sum(x ** 2, axis=1)) - np.exp(np.sum(np.cos(c*x), axis=1) / d) + a + np.exp(1))