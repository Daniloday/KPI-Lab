import os
import sys
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


def optimization(func_0, x_0, y_0, tau, eps):
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

    x = [x_0, 0]
    y = [y_0, 0]

    x_list = [x_0]
    y_list = [y_0]

    point = ax.scatter(x[0], y[0], func["f"](x[0], y[0]), c="red")

    dfdx = func["dfdx"](x[0], y[0])
    dfdy = func["dfdy"](x[0], y[0])

    g_x = [-dfdx, 0]
    g_y = [-dfdy, 0]

    h_x = -dfdx
    h_y = -dfdy

    k = 0

    while True:
        steps = []
        func_values = []

        for i in np.arange(0.01, 0.5, 0.01):
            steps.append(i)
            func_values.append(func["f"](x[0] + i * h_x, y[0] + i * h_y))

        step = steps[func_values.index(min(func_values))]

        x[1] = x[0]
        y[1] = y[0]

        x[0] = x[1] + step * h_x
        y[0] = y[1] + step * h_y

        x_list.append(x[0])
        y_list.append(y[0])

        print("x: ", x[0], "y: ", y[0], "f: ", func["f"](x[0], y[0]))

        ax.scatter(x[0], y[0], func["f"](x[0], y[0]), c="black")

        ax_heatmap.scatter(x_list, y_list, c="black")

        plt.plot(x_list, y_list, c="black")

        dfdx = func["dfdx"](x[0], y[0])
        dfdy = func["dfdy"](x[0], y[0])

        if func_0 == "Ackley":
            if np.sqrt(x[0]**2 + y[0]**2) < eps:
                break
        if func_0 == "Quadratic":
            if np.sqrt((2.5 - x[0])**2 + (1.3 - y[0])**2) < eps:
                break
        if func_0 == "Multimodal":
            if np.sqrt(x[0]**2 + y[0]**2) < eps:
                break

        g_x[1] = g_x[0]
        g_y[1] = g_y[0]

        g_x[0] = -dfdx
        g_y[0] = -dfdy

        beta = ((k + 1) / tau) * (
            (g_x[0] - g_x[1]) * g_x[0] +
            (g_y[0] - g_y[1]) * g_y[0]) / (g_x[0]**2 + g_y[0]**2)

        h_x = g_x[0] + beta * h_x
        h_y = g_y[0] + beta * h_y

        k = k + 1

        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.ioff()

    ax.scatter(x[0], y[0], func["f"](x[0], y[0]), c="blue")
    plt.show()

    plt.plot(x_list, y_list, lw=5, color=black, linestyle='solid')
    plt.show()