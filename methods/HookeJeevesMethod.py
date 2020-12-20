import os
import sys
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


def search(func, x, y, h_x, h_y):
    x_temp = x
    y_temp = y
    f = func["f"](x_temp, y_temp)
    f_x_plus = func["f"](x_temp + h_x, y_temp)
    f_x_minus = func["f"](x_temp - h_x, y_temp)
    f_y_plus = func["f"](x_temp, y_temp + h_x)
    f_y_minus = func["f"](x_temp, y_temp - h_x)

    if f_x_plus < f and f_x_plus <= f_x_minus:
        x_temp = x_temp + h_x
    else:
        if f_x_minus < f and f_x_minus <= f_x_plus:
            x_temp = x_temp - h_x
        else:
            x_temp = x
    f_x_value = func["f"](x_temp, y)

    if f_y_plus < f and f_y_plus <= f_y_minus:
        y_temp = y_temp + h_y
    else:
        if f_y_minus < f and f_y_minus <= f_y_plus:
            y_temp = y_temp - h_y
        else:
            y_temp = y
    f_y_value = func["f"](x, y_temp)

    return [x_temp, y_temp, f_x_value, f_y_value]


def research(func, x_0, y_0, h_x0, h_y0, eps):
    x = x_0
    y = y_0
    h_x = h_x0
    h_y = h_y0

    search_result = search(func, x, y, h_x, h_y)

    while True:
        check = np.sqrt((search_result[0] - x)**2 + (search_result[1] - y)**2)

        if ((search_result[0] != x) or (search_result[1] != y)):
            if (check < eps):
                return [
                    search_result[0], search_result[1],
                    func["f"](search_result[0],
                              search_result[1]), h_x, h_y, True
                ]
            else:
                x = x + 0.75 * (search_result[0] - x)
                y = y + 0.75 * (search_result[1] - y)

                return [x, y, func["f"](x, y), h_x, h_y, False]
        else:
            h_x = h_x / 2
            h_y = h_y / 2
            search_result = search(func, x, y, h_x, h_y)


def optimization(func_0, x_0, y_0, h_x0, h_y0, eps):
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

    x = x_0
    y = y_0

    x_list = [x_0]
    y_list = [y_0]

    ax.scatter(x, y, func["f"](x, y), c="red")
    ax_heatmap.scatter(x_list, y_list, c="black")

    plt.plot(x_list, y_list, c="black")

    h_x = h_x0
    h_y = h_y0

    while True:
        print("x: ", x, "y: ", y, "f: ", func["f"](x, y))
        res = research(func, x, y, h_x, h_y, eps)

        x = res[0]
        x_list.append(x)

        y = res[1]
        y_list.append(y)

        h_x = res[3]
        h_y = res[4]

        ax.scatter(x, y, res[2], c="black")
        ax_heatmap.scatter(x_list, y_list, c="black")

        plt.plot(x_list, y_list, c="black")

        fig.canvas.draw()
        fig.canvas.flush_events()

        if np.sqrt(res[5]) == True:
            break

    plt.ioff()

    print("-----final------")
    print("x: ", x, "y: ", y, "f: ", func["f"](x, y))
    ax.scatter(x, y, func["f"](x, y), c="blue")
    plt.show()