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
    """Load the data and do all standard processing (replace_nan, zscores ...)."""
    series = read_csv(file_name, header=None, delimiter="\t", names=["y"])

    temp_y = series["y"]

    # Fixing NaNs (note the first two cases should not happen)
    if np.isnan(temp_y.iloc[-1]):
        for w in reversed(range(len(temp_y))):
            if not np.isnan(temp_y[w]):
                temp_y[-1] = temp_y[w]
                break
    if np.isnan(temp_y[0]):
        for w in range(len(temp_y)):
            if not np.isnan(temp_y[w]):
                temp_y[-1] = temp_y[w]
                break
    # Interpolate any remaining NaNs
    if temp_y.isna().any():
        np.interp(np.arange(len(temp_y)), 
                  np.arange(len(temp_y))[np.isnan(temp_y) == False], 
                  temp_y[np.isnan(temp_y) == False])

    zrating = stats.zscore(temp_y)

    if max(zrating) > max_zscore or min(zrating) < -max_zscore:
        excluded[1] += 1
    elif np.isnan(sum(temp_y)) or np.std(temp_y) == 0:
        excluded[0] += 1
    elif not np.isnan(sum(temp_y)):
        if group.size == 0:
            group = temp_y
        else:
            group = np.hstack((group, temp_y))
    else:
        raise Exception("ALERT")

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
