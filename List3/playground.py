import numpy as np
from utils import plot_objective, plot_history
from objective_functions import ackley
from evolutionary_strategies import ESPlus

if __name__ == '__main__':
    #plot_objective(ackley, -32.768, 32.768)

    d = 50
    K = 1000.
    esplus = ESPlus(ackley, 10, 32.768)
    tau = K / np.sqrt(2*d)
    tau0 = K / np.sqrt(2*np.sqrt(d))
    res = esplus.optimize(300000, 500000, tau, tau0, 250)
    print(res['result'])
    plot_history(res)