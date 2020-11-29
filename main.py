import os
import sys
import numpy as np

sys.path.append(os.path.abspath('./methods'))

import ConjugateGradientMethod as cgm
import StepByStepGradientDescent as sbsgd
import SteepestGradientDescent as sgd
import ClonAlg as cla
import NutonMethod as nm
import HookeJeevesMethod as hjm
import NelderMeadMethod as nmm


def main():
    x_0 = [[-4.3, -3.9], [3.1, 4.7], [6.3, -0.7]]

    # sgd.optimization("Ackley", -3.5, -4.5, 0.1)
    # sbsgd.optimization("Ackley", -3.5, -4, 0.1, 4, 0.25)
    # cgm.optimization("Ackley", -3.5, -4, 3, 0.05)
    # cla.optimization("Ackley", 0.0)
    # nm.optimization("Ackley", 2.5, -3.1, 0.1)
    # hjm.optimization("Ackley", -4.37, 2.49, 1, 1, 0.01)
    # nmm.optimization("Ackley", x_0, 1, 0.5, 2, 0.1)

    y_0 = [[-4, -4], [3, 2], [5, 1]]
    # sgd.optimization("Quadratic", 0, -np.sqrt(5), 0.3)
    # sbsgd.optimization("Quadratic", 0, -np.sqrt(5), 0.3, 2, 0.25)
    # cgm.optimization("Quadratic", 0, -np.sqrt(5), 3, 0.3)
    # nm.optimization("Quadratic", -4, -3, 0.1)
    # hjm.optimization("Quadratic", -4, -4, 1, 1, 0.01)
    nmm.optimization("Quadratic", y_0, 1, 0.5, 2, 0.1)

    z_0 = [[2.5, 3.7], [5.0, 3.5], [3.5, 4.5]]
    # sgd.optimization("Multimodal", -1, 3, 0.1)
    # cgm.optimization("Multimodal", 4, 2, 8, 0.3)
    # cla.optimization("Multimodal", 0.0)
    # nm.optimization("Multimodal", 3.5, -2.5, 0.1)
    # hjm.optimization("Multimodal", 1, -3, 0.7, 0.7, 0.001)
    # nmm.optimization("Multimodal", z_0, 1, 0.5, 2, 0.1)


if __name__ == "__main__":
    main()
