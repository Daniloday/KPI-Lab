import numpy as np
import random as rd
import matplotlib.pyplot as plt
from numpy import longdouble
import random as rd


def F1(X):  # Первая целевая функция
    return X[0] + 2 * X[1] ** 2 + np.exp(X[0] ** 2 + X[1] ** 2)


def F2(X):  # Вторая целевая функция
    return 100 * (X[1] - X[0] ** 2) ** 2 + (X[0] - 1) ** 2


def F3(X):  # Третья целевая функция (ненужная, но вот да)
    return X[0] * np.sin(4 * np.pi * X[0]) - X[1] * np.sin(4 * np.pi * X[1] + np.pi) + 1


def grad(F, X, Eps):  # Градиент для оптимизационных функцй
    res = np.copy(X)
    for i in range(X.size):
        XRight = np.copy(X)
        XRight[i] += Eps
        XLeft = np.copy(X)
        XLeft[i] -= Eps
        res[i] = (F(XRight) - F(XLeft)) / (2 * Eps)
    return res


def Gessian(F, X, Eps):  # Матрица вторых производных для оптимизационных функций
    res = np.zeros((X.size, X.size))
    for i in range(X.size):
        for j in range(X.size):
            XRR = np.copy(X)
            XRL = np.copy(X)
            XLR = np.copy(X)
            XLL = np.copy(X)
            XRR[i] += Eps
            XRR[j] += Eps
            XRL[i] += Eps
            XRL[j] -= Eps
            XLR[i] -= Eps
            XLR[j] += Eps
            XLL[i] -= Eps
            XLL[j] -= Eps
            res[i, j] = (F(XRR) - F(XRL) - F(XLR) + F(XLL)) / (4 * Eps ** 2)
    return res


def Newton(F, X0, Eps):  # Метод Ньютона (оптимизаци)
    X = np.copy(X0)
    Golden = (1 + 5 ** 0.5) / 2
    Iterations = np.array([X0])
    dx = 1
    while dx > Eps:
        X0 = np.copy(X)
        a = 0.
        b = 10.
        H = np.linalg.inv(Gessian(F, X, Eps)).dot(grad(F, X, Eps))
        while b - a > Eps:
            x1 = b - (b - a) / Golden
            x2 = a + (b - a) / Golden
            if F(X - x1 * H) > F(X - x2 * H):
                a = x1
            else:
                b = x2
        X -= 0.5 * (a + b) * H
        Iterations = np.vstack((Iterations, X))
        dx = np.linalg.norm(X - X0)
    return Iterations


# Далее идут 2 функции, которые делают одно и то же, однако с небольшой поправкой. Дело в том, что
# в методе штрафов в случае барьерных штрафов выход за допустимую область приводит к фиаско в
# работе алгоритма. Потому нужно было добавить некоторую функцию, которая не выпускает точки за пределы
# множества допустимого. То есть как это работает: алгоритм оптимизации действительно действует внутри
# множества, однако он рассматривает только те точки, которые находятся внутри этого множества. По-другому
# никак, потому что в отличии от метода внешних штрафов за пределами области нет никаких штрафов, нужно
# ограничить работу метода

def NewtonKontorovichCorrect(F, X0, Eps, Correct):
    X = np.copy(X0)
    Golden = (1 + 5 ** 0.5) / 2
    Iterations = np.array([X0])
    i = 0
    dx = 1
    while dx > Eps:
        i += 1
        X0 = np.copy(X)
        A = Gessian(F, X, Eps)
        B = grad(F, X, Eps)
        for i in range(X.size):
            for j in range(i + 1, X.size):
                B[j] = B[j] - B[i] * A[j, i] / A[i, i]
                A[j] -= np.copy(A[i] * A[j, i] / A[i, i])
        for i in range(X.size):
            for j in range(i + 1, X.size):
                B[X.size - 1 - j] = B[X.size - 1 - j] - B[X.size - 1 - i] * A[X.size - 1 - j, X.size - 1 - i] / A[
                    X.size - 1 - i, X.size - 1 - i]
                A[X.size - 1 - j] = A[X.size - 1 - j] - A[X.size - 1 - i] * A[X.size - 1 - j, X.size - 1 - i] / A[
                    X.size - 1 - i, X.size - 1 - i]
        for i in range(X.size):
            B[i] /= A[i, i]
        a = 0.
        b = 1.
        B /= np.linalg.norm(B)
        while b - a > Eps:
            x1 = b - (b - a) / Golden
            x2 = a + (b - a) / Golden
            if np.max(Correct(X - x2 * B)) < 0:
                if F(X - x1 * B) > F(X - x2 * B):
                    a = x1
                else:
                    b = x2
            else:
                b = x2
        X -= 0.5 * (a + b) * B
        dx = np.linalg.norm(X - X0)
        Iterations = np.vstack((Iterations, X))
    return Iterations


def NewtonKontorovich(F, X0, Eps):
    X = np.copy(X0)
    Golden = (1 + 5 ** 0.5) / 2
    Iterations = np.array([X0])
    i = 0
    dx = 1
    while dx > Eps:
        i += 1
        X0 = np.copy(X)
        A = Gessian(F, X, Eps)
        B = grad(F, X, Eps)
        for i in range(X.size):
            for j in range(i + 1, X.size):
                B[j] = B[j] - B[i] * A[j, i] / A[i, i]
                A[j] -= np.copy(A[i] * A[j, i] / A[i, i])
        for i in range(X.size):
            for j in range(i + 1, X.size):
                B[X.size - 1 - j] = B[X.size - 1 - j] - B[X.size - 1 - i] * A[X.size - 1 - j, X.size - 1 - i] / A[
                    X.size - 1 - i, X.size - 1 - i]
                A[X.size - 1 - j] = A[X.size - 1 - j] - A[X.size - 1 - i] * A[X.size - 1 - j, X.size - 1 - i] / A[
                    X.size - 1 - i, X.size - 1 - i]
        for i in range(X.size):
            B[i] /= A[i, i]
        a = 0.
        b = 1.
        B /= np.linalg.norm(B)
        while b - a > Eps:
            x1 = b - (b - a) / Golden
            x2 = a + (b - a) / Golden
            if F(X - x1 * B) > F(X - x2 * B):
                a = x1
            else:
                b = x2
        X -= 0.5 * (a + b) * B
        dx = np.linalg.norm(X - X0)
        Iterations = np.vstack((Iterations, X))
    return Iterations


# Внешние штрафы

def ExternalFines(Function, EqualCondition, InequalCondition, StartPoint):
    for FinePower in range(0, 4):
        Fine = 10. ** (2 * FinePower)  # Штраф

        def Fitness(X):  # Поправленная целевая функция без условий
            return Function(X) + Fine * (EqualCondition(X) + InequalCondition(X))

        print("r^k=", Fine)
        Iterations = Newton(Fitness, StartPoint, 0.0001)
        plt.plot(Iterations[:, 0], Iterations[:, 1], color='red')  # график итераций
        plt.show()
        print(Iterations)


# Внутренние штрафы

def InternalFines(Function, EqualCondition, InequalCondition, StartPoint):
    for FinePower in range(0, 4):
        Fine = 10. ** (-2 * FinePower)  # штраф

        def Fitness(X):  # Поправленная целевая функция без условий
            res = 0
            for x in InequalCondition(X):
                res += 1 / x
            res *= -Fine
            return res + Function(X)

        print("r^k=", Fine)
        Iterations = NewtonKontorovichCorrect(Fitness, StartPoint, 0.000001, InequalCondition)
        plt.plot(Iterations[:, 0], Iterations[:, 1], color='blue')  # график итераций
        plt.show()
        print(Iterations)


# Ограничения для внешних штрафов. Всё происходит в кругу, в котором не находятся оптимальные
# точки всех функций, потому ответ будет лежать на границе области. Потому нет необходимости
# добавлять ограничения-равенства, но при желании можно

def ExternalEqualCondition(X):
    return 0


def ExternalInequalCondition(X):
    return np.max([0, (X[0] - 1) ** 2 + (X[1] + 1) ** 2 - 1]) ** 2


# Ограничения для внутренних штрафов. Всё происходит в кругу, в котором не находятся оптимальные
# точки всех функций, потому ответ будет лежать на границе области. Потому нет необходимости
# добавлять ограничения-равенства, но при желании можно

def InternalEqualCondition(X):
    return 0


def InternalInequalCondition(X):
    return np.array([(X[0] - 1) ** 2 + (X[1] + 1) ** 2 - 1])


####################### пример вызова функций
# F1, внешние штрафы
ExternalFines(F1, ExternalEqualCondition, ExternalInequalCondition, np.array([1, -1], dtype=longdouble))
# F1, внутренние штрафы
InternalFines(F1, InternalEqualCondition, InternalInequalCondition, np.array([1, -1], dtype=longdouble))
