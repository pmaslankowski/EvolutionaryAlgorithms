from utils import run_test

if __name__ == '__main__':
    params = {'function_name': 'schwefel',
              'mu': 100000,
              'lam': 200000,
              'd': 10,
              'K': 1,
              'max_iters': 1000,
              'mode': 'comma'}

    run_test(**params)