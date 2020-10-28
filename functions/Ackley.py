import numpy as np


def f(x, y):
    return (-20 * np.e**(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) -
            np.e**(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) +
            np.e + 20)


def dfdx(x, y):
    return np.pi * np.sin(2 * np.pi * x) * np.e**(
        0.5 *
        (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + 2 * x * np.e**(
            -0.2 * np.sqrt(0.5 * (x**2 + y**2))) / np.sqrt(0.5 * (x**2 + y**2))


def dfdy(x, y):
    return np.pi * np.sin(2 * np.pi * y) * np.e**(
        0.5 *
        (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + 2 * y * np.e**(
            -0.2 * np.sqrt(0.5 * (x**2 + y**2))) / np.sqrt(0.5 * (x**2 + y**2))
