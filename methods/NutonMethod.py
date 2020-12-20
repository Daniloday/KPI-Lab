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
    "dfdxdy": Ackley.dfdxdy,
    "dfdydx": Ackley.dfdydx,
    "dfdydy": Ackley.dfdydy
}

arrMultimodal = {
    "f": Multimodal.f,
    "dfdx": Multimodal.dfdx,
    "dfdy": Multimodal.dfdy,
    "dfdxdx": Multimodal.dfdxdx,
    "dfdxdy": Multimodal.dfdxdy,
    "dfdydx": Multimodal.dfdydx,
    "dfdydy": Multimodal.dfdydy
}

arrQuadratic = {
    "f": Quadratic.f,
    "dfdx": Quadratic.dfdx,
    "dfdy": Quadratic.dfdy,
    "dfdxdx": Quadratic.dfdxdx,
    "dfdxdy": Quadratic.dfdxdy,
    "dfdydx": Quadratic.dfdydx,
    "dfdydy": Quadratic.dfdydy
}

arrFunc = {
    "Ackley": arrAckley,
    "Quadratic": arrQuadratic,
    "Multimodal": arrMultimodal
}


def optimization(func_0, x_0, y_0, eps):
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

    grad_x = func["dfdx"](x, y)
    grad_y = func["dfdy"](x, y)

    while np.linalg.norm([grad_x, grad_y]) > eps:
        print("x: ", x, "y: ", y, "f: ", func["f"](x, y))

        H = np.linalg.inv([[func["dfdxdx"](x, y), func["dfdxdy"](x, y)],
                           [func["dfdydx"](x, y), func["dfdydy"](x, y)]])

        p_x = H[0][0] * grad_x + H[0][1] * grad_y
        p_y = H[1][0] * grad_x + H[1][1] * grad_y

        x = x - p_x
        y = y - p_y

        grad_x = func["dfdx"](x, y)
        grad_y = func["dfdy"](x, y)

        x_list.append(x)
        y_list.append(y)

        ax.scatter(x, y, func["f"](x, y), c="black")
        ax_heatmap.scatter(x_list, y_list, c="black")

        plt.plot(x_list, y_list, c="black")

        fig.canvas.draw()
        fig.canvas.flush_events()

    # while True:
    #     dfdx = func["dfdx"](x, y)
    #     dfdy = func["dfdy"](x, y)
    #     dfdxdx = func["dfdxdx"](x, y)
    #     dfdydy = func["dfdydy"](x, y)

    #     h_x = (-dfdx) / dfdxdx
    #     h_y = (-dfdy) / dfdydy

    #     # print(x, y)
    #     # print(dfdx, dfdy)

    #     # if func_0 == "Ackley":
    #     #     if np.sqrt(x**2 + y**2) < eps:
    #     #         break
    #     # if func_0 == "Quadratic":
    #     #     if np.sqrt((2.5 - x)**2 + (1.3 - y)**2) < eps:
    #     #         break
    #     # if func_0 == "Multimodal":
    #     #     if np.sqrt(x**2 + y**2) < eps:
    #     #         break

    #     print("x : ", x, "y : ", y)

    #     x = x + h_x
    #     y = y + h_y

    #     x_list.append(x)
    #     y_list.append(y)

    #     ax.scatter(x, y, func["f"](x, y), c="black")
    #     ax_heatmap.scatter(x_list, y_list, c="black")

    #     plt.plot(x_list, y_list, c="black")

    #     fig.canvas.draw()
    #     fig.canvas.flush_events()

    plt.ioff()

    print("-----final------")
    print("x: ", x, "y: ", y, "f: ", func["f"](x, y))
    ax.scatter(x, y, func["f"](x, y), c="blue")
    plt.show()