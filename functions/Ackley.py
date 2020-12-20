import numpy as np


def f(x, y):
    return (-20 * np.e ** (-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) -
            np.e ** (0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) +
            np.e + 20)


def dfdx(x, y):
    return np.pi * np.sin(2 * np.pi * x) * np.e ** (
            0.5 *
            (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + 2 * x * np.e ** (
                   -0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) / np.sqrt(0.5 * (x ** 2 + y ** 2))


def dfdy(x, y):
    return np.pi * np.sin(2 * np.pi * y) * np.e ** (
            0.5 *
            (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + 2 * y * np.e ** (
                   -0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) / np.sqrt(0.5 * (x ** 2 + y ** 2))


def dfdxdx(x, y):
    return 2 * np.pi * np.pi * np.cos(2 * np.pi * x) * np.e ** (
            0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + (
            2 * np.e ** (-0.2 * np.sqrt(0.5 * (x * x + y * y)))) / (np.sqrt(0.5 * (x * x + y * y))) - (
            np.pi * np.pi * np.sin((2 * np.pi * x) ** 2) * np.e ** (
            0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) - (
            x * x * np.e ** (-0.2 * np.sqrt(0.5 * (x * x + y * y)))) / ((0.5 * (x * x + y * y)) ** (1.5)) - (
            0.2 * x * x * np.e ** (-0.2 * np.sqrt(0.5 * (x * x + y * y)))) / (0.5 * (x * x + y * y)))

def dfdxdy(x, y):
    return -(x * y * np.e ** (-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))))/((np.sqrt(0.5 * (x ** 2 + y ** 2))) ** 3) - (np.pi ** 2) * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) * (np.e ** (0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))) - 0.2 * x * y * (-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) / (0.5 * (x ** 2 + y ** 2))


def dfdydx(x, y):
    return -(x * y * np.e**(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))) / (
        (np.sqrt(0.5 * (x**2 + y**2)))**3) - (np.pi**2) * np.sin(
            2 * np.pi * x) * np.sin(2 * np.pi * y) * (
                np.e**(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
            ) - 0.2 * x * y * (-0.2 * np.sqrt(0.5 *
                                              (x**2 + y**2))) / (0.5 *
                                                                 (x**2 + y**2))


def dfdydy(x, y):
    return 2 * np.pi * np.pi * np.cos(2 * np.pi * y) * np.e ** (
            0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + (
            2 * np.e ** (-0.2 * np.sqrt(0.5 * (x * x + y * y)))) / (np.sqrt(0.5 * (x * x + y * y))) - (
            np.pi * np.pi * np.sin((2 * np.pi * y) ** 2) * np.e ** (
            0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))) - (
            y * y * np.e ** (-0.2 * np.sqrt(0.5 * (x * x + y * y)))) / ((0.5 * (x * x + y * y)) ** (1.5)) - (
            0.2 * y * y * np.e ** (-0.2 * np.sqrt(0.5 * (x * x + y * y)))) / (0.5 * (x * x + y * y))
