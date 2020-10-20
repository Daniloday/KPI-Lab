import numpy as np

def f(x, y):
    return x * np.sin(4 * np.pi * x) - y * np.sin(4 * np.pi * y + np.pi) + 1

def dfdx(x, y):
	return np.sin(4 * np.pi * x) + x * np.cos(4 * np.pi * x) * 4 * np.pi

def dfdy(x, y):
	return np.sin(4 * np.pi * y) + y * np.cos(4 * np.pi * y) * 4 * np.pi