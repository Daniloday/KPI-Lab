import numpy as np

def f(x, y):
    return x * np.sin(4 * np.pi * x) - y * np.sin(4 * np.pi * y + np.pi) + 1