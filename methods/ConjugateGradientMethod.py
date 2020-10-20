import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.abspath('./functions'))

import Ackley
import Multimodal
import Quadratic

arrAckley = {"f": Ackley.f, "dfdx": Ackley.dfdx, "dfdy": Ackley.dfdy}
arrMultimodal = {"f": Multimodal.f, "dfdx": Multimodal.dfdx, "dfdy": Multimodal.dfdy}
arrQuadratic = {"f": Quadratic.f, "dfdx": Quadratic.dfdx, "dfdy": Quadratic.dfdy}

arrFunc = {"Ackley": arrAckley, "Quadratic": arrQuadratic, "Multimodal": arrMultimodal}

def optimization(func, x, y, eps):
    func = arrFunc[func]

    x_plt = np.arange(-30.0, 30.0, 0.1)
    y_plt = np.arange(-30.0, 30.0, 0.1)

    f_plt = np.array([[func["f"](x, y) for x in x_plt] for y in y_plt])

    plt.ion()
    fig = plt.figure()
    ax = Axes3D(fig)

    ox, oy = np.meshgrid(x_plt, y_plt)
    ax.plot_surface(ox, oy, f_plt, color="y", alpha=0.5)

    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_zlabel("f")

    point = ax.scatter(x, y, func["f"](x, y), c="red")

    while True:
        dfdx = func["dfdx"](x, y)
        dfdy = func["dfdy"](x, y)

        module = np.sqrt(dfdx ** 2 + dfdy ** 2)

        # print(x, y)
        # print(dfdx, dfdy)

        if module < eps or np.isnan(dfdx) or np.isnan(dfdy):
            break

        h_x = -np.sign(dfdx)
        h_y = -np.sign(dfdy)

        steps = []
        func_values = []

        for i in np.arange(0.50, 2.01, 0.01):
            steps.append(i)
            func_values.append(func["f"](x + i * h_x, y + i * h_y))

        step = steps[func_values.index(min(func_values))]

        x = x + step * h_x
        y = y + step * h_y

        print(x, y)

        ax.scatter(x, y, func["f"](x, y), c = "red")

        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.ioff()

    print(x, y)
    print("kek")
    ax.scatter(x, y, func["f"](x, y), c = "blue")
    plt.show()