import os
import sys

sys.path.append(os.path.abspath('./methods'))

import SteepestGradientDescent as sgd
import StepByStepGradientDescent as sbsgd

def main():
    # sgd.optimization("Ackley", 2.5, 2.5, 0.1)
    sbsgd.optimization("Ackley", 3.0, 2.0, 0.01, 5.0, 0.7)

if __name__ == "__main__":
    main()