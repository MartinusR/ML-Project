
"""
Utility functions
"""

import numpy as np


def set_all_args(obj, argdict):
    for k in argdict.keys():
        if hasattr(obj, k):
            setattr(obj, k, argdict[k])
        else:
            print("Warning: parameter name {} not found!".format(k))

def div0(a,b):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c = np.nan_to_num(c)
    return c
