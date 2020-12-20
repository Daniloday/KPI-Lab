import os
import sys
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from heapq import nsmallest, nlargest

sys.path.append(os.path.abspath('./functions'))

import Ackley
import Multimodal
import Quadratic

arrAckley = {
    "f": Ackley.f,
    "dfdx": Ackley.dfdx,
    "dfdy": Ackley.dfdy,
    "dfdxdx": Ackley.dfdxdx,
    "dfdydy": Ackley.dfdydy
}

arrMultimodal = {
    "f": Multimodal.f,
    "dfdx": Multimodal.dfdx,
    "dfdy": Multimodal.dfdy,
    "dfdxdx": Multimodal.dfdxdx,
    "dfdydy": Multimodal.dfdydy
}

arrQuadratic = {
    "f": Quadratic.f,
    "dfdx": Quadratic.dfdx,
    "dfdy": Quadratic.dfdy,
    "dfdxdx": Quadratic.dfdxdx,
    "dfdydy": Quadratic.dfdydy
}

arrFunc = {
    "Ackley": arrAckley,
    "Quadratic": arrQuadratic,
    "Multimodal": arrMultimodal
}

x = 0
y = 0


def optimization(func_0, x_0, alpha, beta, gamma, epsilon):
    func = arrFunc[func_0]

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

    x = 0
    y = 0

    x_list = []
    y_list = []

    for x in x_0:
        point = ax.scatter(x[0], x[1], func["f"](x[0], x[1]), c="red")
        x_list.append(x[0])
        y_list.append(x[1])

    x_plot = x_list + [x_list[0]]
    y_plot = y_list + [y_list[0]]

    plt.plot(x_plot, y_plot)

    # paths = ax_heatmap.scatter(x_list, y_list, c="black")
    # lines = plt.plot(x_list, y_list, c="black")

    iterations = 0

    while True:
        print("x: ", x_list, "y: ", y_list)

        func_values = []

        for point in x_0:
            func_values.append(func["f"](point[0], point[1]))

        min_values = nsmallest(2, func_values)

        x_min = x_0[func_values.index(min_values[0])][0]
        y_min = x_0[func_values.index(min_values[0])][1]

        x_submin = x_0[func_values.index(min_values[1])][0]
        y_submin = x_0[func_values.index(min_values[1])][1]

        max_values = nlargest(2, func_values)

        x_max = x_0[func_values.index(max_values[0])][0]
        y_max = x_0[func_values.index(max_values[0])][1]

        x_submax = x_0[func_values.index(max_values[1])][0]
        y_submax = x_0[func_values.index(max_values[1])][1]

        x_center = 0
        y_center = 0

        for point in x_0:
            x_center = x_center + point[0] - x_max
            y_center = y_center + point[1] - y_max

        x_center = x_center / (len(x_0) - 1)
        y_center = y_center / (len(x_0) - 1)

        sigma = 0

        for point in x_0:
            sigma = sigma + (func["f"](point[0], point[1]) -
                             func["f"](x_center, y_center))**2

        sigma = np.sqrt(sigma / (len(x_0)))

        if (sigma <= epsilon):
            x = x_min
            y = y_min

            break

        x_mirror = x_center + alpha * (x_center - x_max)
        y_mirror = y_center + alpha * (y_center - y_max)

        if func["f"](x_mirror, y_mirror) < func["f"](x_min, y_min):
            x_four = x_center + gamma * (x_mirror - x_center)
            y_four = y_center + gamma * (y_mirror - y_center)

            if func["f"](x_four, y_four) < func["f"](x_min, y_min):
                x_0[func_values.index(max_values[0])][0] = x_four
                x_0[func_values.index(max_values[0])][1] = y_four

                x_list[x_list.index(x_max)] = x_four
                y_list[y_list.index(y_max)] = y_four

            elif func["f"](x_four, y_four) >= func["f"](x_min, y_min):
                x_0[func_values.index(max_values[0])][0] = x_mirror
                x_0[func_values.index(max_values[0])][1] = y_mirror

                x_list[x_list.index(x_max)] = x_mirror
                y_list[y_list.index(y_max)] = y_mirror

        elif (func["f"](x_mirror, y_mirror) > func["f"](x_submax, y_submax)
              ) and (func["f"](x_mirror, y_mirror) <= func["f"](x_max, y_max)):
            x_0[func_values.index(
                max_values[0])][0] = x_center + beta * (x_max - x_center)
            x_0[func_values.index(
                max_values[0])][1] = y_center + beta * (y_max - y_center)

            x_list[x_list.index(x_max)] = x_0[func_values.index(
                max_values[0])][0]
            y_list[y_list.index(y_max)] = x_0[func_values.index(
                max_values[0])][1]

        elif (func["f"](x_mirror, y_mirror) > func["f"](
                x_min, y_min)) and (func["f"](x_mirror, y_mirror) <= func["f"](
                    x_submax, y_submax)):
            x_0[func_values.index(max_values[0])][0] = x_center
            x_0[func_values.index(max_values[0])][1] = y_center

            x_list[x_list.index(x_max)] = x_center
            y_list[y_list.index(y_max)] = y_center

        elif func["f"](x_mirror, y_mirror) > func["f"](x_max, y_max):
            x_list.clear()
            y_list.clear()

            for point in x_0:
                point[0] = x_min + 0.5 * (point[0] - x_min)
                x_list.append(point[0])

                point[1] = y_min + 0.5 * (point[1] - y_min)
                y_list.append(point[1])

        iterations = iterations + 1

        ax.scatter(x_list, y_list, c="black")

        # paths.set_visible(False)
        # line = lines.pop(0)
        # line.remove()

        # paths = ax_heatmap.scatter(x_list, y_list, c="black")

        x_plot = x_list + [x_list[0]]
        y_plot = y_list + [y_list[0]]

        plt.plot(x_plot, y_plot)

        # lines = plt.plot(x_list, y_list, c="black")

        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.ioff()

    print("-----final------")
    print("x: ", x_list, "y: ", y_list, "f: ")
    ax.scatter(x, y, func["f"](x, y), c="blue")
    plt.show()