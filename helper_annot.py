#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:00:38 2023

@author: elliemorgenroth
"""
# Get some useful packages loaded
import numpy as np
from pandas import read_csv
from scipy import stats


def load_data(file_name, max_zscore, group, excluded):
    """
    Load the data and do all standard processing (smoothing, replace_nan, zscores ...).
    """
    series = read_csv(file_name, header=None, delimiter="\t", names=["y"])

    temp_y = series["y"]
    temp_y = temp_y.fillna(-1)
    for v in range(len(temp_y) - 1):
        if temp_y[v] == -1 and v > 1:
            w = 1
            while temp_y[v] == -1 and temp_y[w + v] == -1:
                w += 1
            temp_y[v] = (temp_y[v - 1] + temp_y[v + w]) / (w + 1)

    zrating = stats.zscore(temp_y)

    if sum(temp_y) == -1 or np.std(temp_y) == 0:
        excluded[0] += 1

    elif max(zrating) > max_zscore and min(zrating) > -max_zscore:
        excluded[1] += 1

    elif sum(temp_y) != -1 and max(zrating) < max_zscore:
        if group.size == 0:
            group = temp_y
        else:
            group = np.hstack((group, temp_y))
    else:
        print("ALERT")

    return group, excluded


def lins_ccc(y_true, y_pred, output="CORR"):
    """
    Compute CCC or correlation.

    CCC = 2 * COVAR[X,Y] / (VAR[X] + VAR[Y] + (E[X] - E[Y])^2)
    """
    t = y_true.mean()
    p = y_pred.mean()
    St = y_true.var()
    Sp = y_pred.var()
    Spt = np.mean((y_true - t) * (y_pred - p))
    if output == "CCC":
        return 2 * Spt / (St + Sp + (t - p) ** 2)
    else:
        return np.corrcoef(y_true, y_pred)[0, 1]