import os
import sys
import numpy as np

sys.path.append(os.path.abspath('./methods'))

import ConjugateGradientMethod as cgm
import StepByStepGradientDescent as sbsgd
import SteepestGradientDescent as sgd
import ClonAlg as cla


def main():
    # sgd.optimization("Ackley", -3.5, -4.5, 0.1)
    # sbsgd.optimization("Ackley", -3.5, -4, 0.1, 4, 0.25)
    # cgm.optimization("Ackley", -3.5, -4, 3, 0.05)
    # cla.optimization("Ackley", 0.0)

    # sgd.optimization("Quadratic", 0, -np.sqrt(5), 0.3)
    # sbsgd.optimization("Quadratic", 0, -np.sqrt(5), 0.3, 2, 0.25)
    cgm.optimization("Quadratic", 0, -np.sqrt(5), 3, 0.3)

    # sgd.optimization("Multimodal", -1, 3, 0.1)
    # cgm.optimization("Multimodal", 4, 2, 8, 0.3)
    # cla.optimization("Multimodal", 0.0)


if __name__ == "__main__":
    main()
