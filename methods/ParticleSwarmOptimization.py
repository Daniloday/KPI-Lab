import os
import sys
import random
import copy
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


def optimization(func_0, parts_count, iteration_count):
    func = arrFunc[func_0]

    x_plt = np.linspace(-20.0, 20.0, 250)
    y_plt = np.linspace(-20.0, 20.0, 250)

    f_plt = np.array([[func["f"](x, y) for x in x_plt] for y in y_plt])
    f_plt = -f_plt

    plt.ion()

    fig = plt.figure()
    ax = Axes3D(fig)

    ox, oy = np.meshgrid(x_plt, y_plt)
    # ax.plot_surface(ox, oy, f_plt, rstride=1, cmap=cm.nipy_spectral, alpha=0.7)
    ax.plot_surface(ox, oy, f_plt, rstride=1, color='c', alpha=0.2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("F")

    x1 = np.linspace(-10, 10, 1000)
    x2 = np.linspace(-10, 10, 1000)
    X, Y = np.meshgrid(x1, x2)
    Z = -(func["f"](X, Y))

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

    trajectories = []
    points = []
    individual_min = []
    individual_min_func = []
    speeds = []
    collective_min = []

    for i in range(0, parts_count):
        x = random.randrange(-500, 500, 1) / 100
        y = random.randrange(-500, 500, 1) / 100
        f = -func["f"](x, y)

        points.append([x, y])
        individual_min.append([x, y])
        individual_min_func.append(f)
        trajectories.append([[x, y]])
        speeds.append([
            random.randrange(-1000, 1000, 1) / 1000,
            random.randrange(-1000, 1000, 1) / 1000
        ])

    collective_min = copy.deepcopy(points[individual_min_func.index(
        max(individual_min_func))])
    collective_min_func = copy.deepcopy(max(individual_min_func))

    for iteration in range(0, parts_count):
        ax.scatter(points[iteration][0],
                   points[iteration][1],
                   individual_min_func[iteration],
                   c="red")

    for tr in trajectories:
        x_list = []
        y_list = []

        for point in tr:
            x_list.append(point[0])
            y_list.append(point[1])

        ax_heatmap.plot(x_list, y_list, marker='o')

    for iteration in range(0, iteration_count):
        print("-----------------")

        for part in range(0, parts_count):
            alpha = random.randrange(1, 1000, 1) / 1000
            beta = 1 - alpha

            speeds[part][0] = speeds[part][0] + alpha * (
                individual_min[part][0] -
                points[part][0]) + beta * (collective_min[0] - points[part][0])

            speeds[part][1] = speeds[part][1] + alpha * (
                individual_min[part][1] -
                points[part][1]) + beta * (collective_min[1] - points[part][1])

            points[part][0] = points[part][0] + speeds[part][0]
            points[part][1] = points[part][1] + speeds[part][1]

            func_value = -(func["f"](points[part][0], points[part][1]))

            ax.scatter(points[part][0], points[part][1], func_value, c="red")

            for tr in trajectories:
                x_list = []
                y_list = []

                for point in tr:
                    x_list.append(point[0])
                    y_list.append(point[1])

                ax_heatmap.plot(x_list, y_list, marker='o')

            # print("func_value: ", func_values[j], "indiviuad[j]: ",
            #       individual_min[j], "min_func", min_func, "x: ",
            #       collective_min[0], "y: ", collective_min[1])

            trajectories[part].append(copy.deepcopy(points[part]))

            if func_value > individual_min_func[part]:
                individual_min[part] = copy.deepcopy(points[part])
                individual_min_func[part] = copy.deepcopy(func_value)

                if func_value > collective_min_func:
                    collective_min = copy.deepcopy(points[part])
                    collective_min_func = copy.deepcopy(func_value)

        # if min(func_values) < func["f"](collective_min[0], collective_min[1]):
        #     collective_min = points[func_values.index(min(func_values))]
        #     min_func = func["f"](collective_min[0], collective_min[1])

        # for i in range(0, parts_count):
        #     ax.scatter(points[i][0], points[i][1], func["f"](points[i][0], points[i][1]), c="red")

        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.ioff()

    print("x: ", collective_min[0], "y: ", collective_min[1], "f: ",
          -collective_min_func)
    ax.scatter(collective_min[0],
               collective_min[1],
               collective_min_func,
               c="blue")
    plt.show()