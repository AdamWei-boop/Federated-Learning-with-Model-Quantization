#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    if isinstance(w[0],np.ndarray) == True:
        for i in range(1, len(w)):
            w_avg += w[i]
        w_avg = w_avg/len(w)
    else:
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg