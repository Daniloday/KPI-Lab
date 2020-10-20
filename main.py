import os
import sys

sys.path.append(os.path.abspath('./methods'))

import SteepestGradientDescent as sgd
import StepByStepGradientDescent as sbsgd
import ConjugateGradientMethod as cgm

def main():
    # sgd.optimization("Ackley", 2.5, 2.5, 0.1)
    # sbsgd.optimization("Ackley", 3.0, 2.0, 0.01, 2.0, 0.5)
    # cgm.optimization("Ackley", 9.43, -4.7, 0.01)

    cgm.optimization("Quadratic", -3, 7, 0.01)

if __name__ == "__main__":
    main()