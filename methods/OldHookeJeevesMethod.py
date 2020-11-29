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

    point = ax.scatter(x, y, func["f"](x, y), c="red")
    ax_heatmap.scatter(x_list, y_list, c="black")

    plt.plot(x_list, y_list, c="black")

    h_x = h_x0
    h_y = h_y0

    iterations = 0

    while True:
        if (h_x < eps and h_y < eps):
            break
        fStart = func["f"](x, y)

        while (h_x > eps):
            xPlus = func["f"](x + h_x, y)
            xMinus = func["f"](x - h_x, y)
            if (xPlus < xMinus):
                if (xPlus < fStart):
                    x = x + h_x
                    print("x = x + h_x worked, x = ", x)
                    break
                else:
                    h_x = 0.5 * h_x
                    print("h_x decreased, h_x = ", h_x)
            else:
                if (xMinus < fStart):
                    x = x - h_x
                    print("x = x - h_x worked, x = ", x)
                    break
                else:
                    h_x = 0.5 * h_x
                    print("h_x decreased 2, h_x = ", h_x)

        fStart = func["f"](x, y)

        x_list.append(x)
        y_list.append(y)

        iterations = iterations + 1

        ax.scatter(x, y, func["f"](x, y), c="black")
        ax_heatmap.scatter(x_list, y_list, c="black")

        plt.plot(x_list, y_list, c="black")

        fig.canvas.draw()
        fig.canvas.flush_events()

        while (h_y > eps):
            yPlus = func["f"](x, y + h_y)
            yMinus = func["f"](x, y - h_y)
            if (yPlus < yMinus):
                if (yPlus < fStart):
                    y = y + h_y
                    break
                else:
                    h_y = 0.5 * h_y
            else:
                if (yMinus < fStart):
                    y = y - h_y
                    break
                else:
                    h_y = 0.5 * h_y

        print("x : ", x, "y : ", y)

        x_list.append(x)
        y_list.append(y)

        iterations = iterations + 1

        ax.scatter(x, y, func["f"](x, y), c="black")
        ax_heatmap.scatter(x_list, y_list, c="black")

        plt.plot(x_list, y_list, c="black")

        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.ioff()

    print("x: ", x, "y: ", y, "f: ", func["f"](x, y))
    print("iterations count = ", iterations)
    ax.scatter(x, y, func["f"](x, y), c="blue")
    plt.show()