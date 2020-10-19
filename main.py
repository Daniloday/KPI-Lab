import os
import sys

sys.path.append(os.path.abspath('./methods'))

import SteepestGradientDescent as sgd

sgd.optimization("Ackley", 2.5, 2.5, 0.1)