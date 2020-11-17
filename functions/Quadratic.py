import numpy as np


def f(x, y):
    return 10 * x**2 - 4 * x * y + 7 * y**2 - 4 * np.sqrt(5) * (5 * x + y) - 16


def dfdx(x, y):
    return 20 * x - 4 * y - 20 * np.sqrt(5)


def dfdy(x, y):
    return -4 * x + 14 * y - 4 * np.sqrt(5)


def dfdxdx(x, y):
    return 20


def dfdydy(x, y):
    return 14