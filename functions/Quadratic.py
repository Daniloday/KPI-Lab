import numpy as np

def f(x, y):
    return 10 * (x ** 2 - y) ** 2 + (x - 1) ** 2

def dfdx(x, y):
    return 2 * (20 * x ** 3 - 20 * x * y + x - 1)

def dfdy(x, y):
    return -20 * (x ** 2 - y)