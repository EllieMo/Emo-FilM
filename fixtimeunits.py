#!/usr/bin/env python3

import sys
from copy import deepcopy as dc
from os.path import isfile

import niutils as niu
import numpy as np

fname = sys.argv[1]

if not isfile(fname):
    raise Exception(f"File {fname} does not exist.")

d, m, i = niu.load_nifti_get_mask(fname)

i_n = dc(i)

i_n._header["xyzt_units"] = np.array(10, dtype="uint8")

niu.export_nifti(d, i_n, fname)
