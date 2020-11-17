import numpy as np


def f(x, y):
    return x**2 + y**2 - np.cos(18 * x) - np.cos(18 * y)


def dfdx(x, y):
    return 2 * x + 18 * np.sin(18 * x)


def dfdy(x, y):
    return 2 * y + 18 * np.sin(18 * y)


def dfdxdx(x, y):
    return 2 + 18 * 18 * np.cos(18 * x)


def dfdydy(x, y):
    return 2 + 18 * 18 * np.cos(18 * y)
