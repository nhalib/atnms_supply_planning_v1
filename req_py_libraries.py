import pandas as pd
import numpy as np
import random

from scipy.stats import binom
from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition
import os
#os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers




non_uniform_weight_factor = 1

np.random.seed(2000)
