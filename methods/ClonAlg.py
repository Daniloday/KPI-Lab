import os
import sys
import random
from random import randint
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.abspath('./functions'))

import Quadratic
import Multimodal
import Ackley

arrAckley = {"f": Ackley.f, "dfdx": Ackley.dfdx, "dfdy": Ackley.dfdy}
arrMultimodal = {
    "f": Multimodal.f,
    "dfdx": Multimodal.dfdx,
    "dfdy": Multimodal.dfdy
}
arrQuadratic = {
    "f": Quadratic.f,
    "dfdx": Quadratic.dfdx,
    "dfdy": Quadratic.dfdy
}

arrFunc = {
    "Ackley": arrAckley,
    "Quadratic": arrQuadratic,
    "Multimodal": arrMultimodal
}


def generateAntibody():
    antibody_x = []
    for i in range(0, 5):
        antibody_x.append(randint(0, 1))

    antibody_y = []
    for i in range(0, 5):
        antibody_y.append(randint(0, 1))

    return [antibody_x, antibody_y]


def clone(generation):
    generation_x = []

    for i in range(0, len(generation)):
        for j in range(0, 5):
            generation_temp = copy.deepcopy(generation[i])
            generation_x = generation_x + [generation_temp]

    return generation_x


def convert(generation):
    generation_ten = []

    for i in range(0, len(generation)):
        x = 0
        y = 0

        for j in range(0, 5):
            x = x + generation[i][0][j] * 2**(4 - j)

        for j in range(0, 5):
            y = y + generation[i][1][j] * 2**(4 - j)

        point = [x, y]

        generation_ten.append(point)

    return generation_ten


def mutate(generation):
    generation_x = generation

    for i in range(0, len(generation_x)):
        for m in range(0, 2):
            for j in range(0, 5):
                rand = randint(1, 100)

                if rand <= 30:
                    if generation[i][m][j] == 0:
                        generation_x[i][m][j] = 1
                    if generation[i][m][j] == 1:
                        generation_x[i][m][j] = 0

    return generation_x


def select(generation, func):
    minimums = []

    functionValues = []

    for i in range(0, len(generation)):
        functionValues.append(func["f"](generation[i][0], generation[i][1]))

    for i in range(0, 10):
        minimums.append(
            functionValues.index(min(functionValues[i * 5:i * 5 + 5])))

    return minimums


def replace(generation_orig, generation_mutate, idx):
    for i in range(0, 10):
        generation_orig[i] = copy.deepcopy(generation_mutate[idx[i]])

    return generation_orig


def modify(generation, generation_ten, func):
    functionValues = []

    for i in range(0, len(generation)):
        functionValues.append(func["f"](generation_ten[i][0],
                                        generation_ten[i][1]))

    maximum = 0

    for i in range(0, 10):
        maximum = functionValues.index(max(functionValues))

    generation[maximum] = generateAntibody()

    return generation


def find_min(generation, generation_ten, func):
    generation_convert = convert(generation)

    functionValues = []

    for i in range(0, len(generation)):
        functionValues.append(func["f"](generation_ten[i][0],
                                        generation_ten[i][1]))

    minimum = 0

    for i in range(0, 10):
        minimum = functionValues.index(min(functionValues))

    return (generation_ten[minimum])


x = 100
y = 100


def optimization(func_0, eps):
    func = arrFunc[func_0]

    x_list = []
    y_list = []

    x_plt = np.linspace(-20.0, 20.0, 250)
    y_plt = np.linspace(-20.0, 20.0, 250)

    f_plt = np.array([[func["f"](x, y) for x in x_plt] for y in y_plt])

    plt.ion()

    fig = plt.figure()
    ax = Axes3D(fig)

    x1 = np.linspace(-5, 5, 400)
    x2 = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(x1, x2)
    Z = func["f"](X, Y)

    ox, oy = np.meshgrid(x_plt, y_plt)
    ax.plot_surface(ox, oy, f_plt, rstride=1, cmap=cm.nipy_spectral, alpha=0.7)

    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_zlabel("f")

    heatmap = plt.figure()
    # heatmap, ax_heat = plt.subplots(nrows=1, ncols=2)
    ax_heatmap = heatmap.add_subplot(1, 1, 1)
    plt.pcolormesh(X, Y, Z, cmap='RdYlGn')
    plt.colorbar()
    plt.contour(X,
                Y,
                Z,
                colors="black",
                alpha=0.5,
                linestyles="dotted",
                linewidth=0.5)
    func = arrFunc[func_0]

    generation = []

    for i in range(0, 10):
        generation.append(generateAntibody())

    while True:
        generation_clone = clone(generation)
        print("Clone")
        print(generation_clone)

        generation_mutate = mutate(generation_clone)
        print("Mutate")
        print(generation_mutate)

        generation_ten = convert(generation_mutate)
        print("Selection")
        print(select(generation_ten, func))

        generation_replace = replace(generation, generation_mutate,
                                     select(generation_ten, func))
        print("Replacing")
        print(generation_replace)

        generation_modify = modify(generation_replace,
                                   convert(generation_replace), func)
        print("Modifying")
        print(generation_modify)

        minimum = find_min(generation_modify, convert(generation_modify), func)

        x = minimum[0]
        y = minimum[1]

        x_list.append(x)
        y_list.append(y)

        print("x: ", x, "y: ", y, "f: ", func["f"](x, y))

        ax.scatter(x, y, func["f"](x, y), c="black")

        ax_heatmap.scatter(x_list, y_list, c="black")

        plt.plot(x_list, y_list, c="black")

        if func_0 == "Ackley":
            if func["f"](x, y) == eps:
                break
        if func_0 == "Multimodal":
            if np.sqrt(x**2 + y**2) == eps:
                break

        generation = generation_modify

        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.ioff()
    print("x: ", x, "y: ", y, "f: ", func["f"](x, y))

    ax.scatter(x, y, func["f"](x, y), c="blue")
    plt.show()