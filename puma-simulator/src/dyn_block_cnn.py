import sys, json

# import dependancy files
import numpy as np
import math
import include.config as cfg
#import include.configTest as cfg
import include.constants as param
import constants_digital as digi_param
import src.ima_modules as imod

from data_convert import *

phy2log_ratio = cfg.phy2log_ratio # ratio of physical to logical xbar
# datamem_off is the start of address sapce of datamemory
datamem_off = cfg.datamem_off # each matrix has 6 memory spaces (1 for f/b, 2 for d)
my_xbar_count = 0

def conv_layer():
    # 1. Matmul

    # 2. Max pooling

    # 3. Relu