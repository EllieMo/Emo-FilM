#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 09:44:15 2022

@author: elliemorgenroth

Fix file paths to your local arrangements before running

This script is everything we did with the validation data returned from the acquisition
1. Arrange data to the length of the films
3. Interpolate to make time course and zscore
4. z-score
5. Average across subjects
5. calculate correlation with consensus annotation
"""
import glob
import json
import os
import sys
import warnings

# Get some useful packages loaded
import numpy as np
import pandas as pd
import scipy
from mat4py import loadmat
from matplotlib import gridspec
from matplotlib import pyplot as plt

from constants_emo_film import ALL_SUBJECTS_FMRI as subs
from constants_emo_film import DURS
from constants_emo_film import ITS_VALIDATION as its
from constants_emo_film import MOVIES

# Paths to where everything is and where results are saved
source = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/Sinergia_Emo/Emo-FilM"
supp_table = (
    sys.argv[2]
    if len(sys.argv) > 2
    else "/Volumes/Sinergia_Emo/EPFL_drive/Sinergia Project/Writing/Data_Paper/Supplementary Tables.xlsx"
)
outdir = (
    sys.argv[3]
    if len(sys.argv) > 3
    else "/Volumes/Sinergia_Emo/EPFL_drive/Sinergia Project/Writing/Data_Paper/Figures"
)
save = os.path.join(source, "fMRIstudy")
main = os.path.join(source, "Annotstudy", "derivatives")
val = os.path.join(source, "fMRIstudy", "derivatives", "validation")

j_file = open(os.path.join(main, "Annot_AfterTheRain_stim.json"))
j_dic = json.load(j_file)

val_times = loadmat(os.path.join(val, "ValTimes.mat"))

# # .mat files are loaded a little strange, which is why it needs to be reshaped in this particular way
val_its = loadmat(os.path.join(val, "ValItems.mat"))
val_its = np.asarray(list(val_its.values())).flatten().reshape(5, 32).transpose()

meta_data = pd.read_excel(supp_table, header=1)

for i in subs:
    try:  # This needs to be like this, as the subject ID is the index
        sidx = int(i[1:])
        items = list(val_its[sidx - 1])
    except KeyError:
        warnings.warn("no matching items for subject")
        continue

    # ## Lists files as returned from our acquisition
    files = glob.glob(
        os.path.join(save, f"sub-{i}", "ses*", "beh", f"sub-{i}_*_task-*_events.tsv")
    )

    # ## Loop Over list of all _val files
    for file in files:
        movie = file.split("_")[-2].split("-")[1]
        movie_idx = MOVIES.index(movie)

        # ## Get time stamps for annotated clips
        vTimes = val_times[movie]

        vali = pd.read_csv(file, delimiter="\t")

        # ## Load validation file

        vali = np.asarray(vali)
        vali = vali[:, 2:7]

        # ## Arrange files to the full length for each item
        for m in range(np.shape(vali)[1]):
            item = vali[:, m]
            itemH = items[m]
            n_times = np.empty(DURS[movie_idx]) * np.nan

            for l in range(len(vTimes)):
                tim = vTimes[l]
                try:
                    n_times[tim[0] : tim[1]] = item[l]
                except:
                    continue
                np.savetxt(
                    os.path.join(val, f"sub-{i}_{movie}_{itemH}.csv"),
                    n_times,
                )

# ## makes continuous time course, z-score within subject and saves data
for i in sorted(its):
    for s in subs:
        files = glob.glob(os.path.join(val, f"sub-{s}_*_{i}.csv"))

        combined = []
        valid_films = []
        for f in sorted(files):
            valid_films.append(f.split("_")[-2])
            s_val = pd.read_csv(f, header=None)
            s_val = s_val.interpolate(method="linear")
            s_val = s_val.to_numpy()

            if len(combined) == 0:
                combined = s_val
            else:
                combined = np.vstack([combined, s_val])

        # ## z-score
        combined = (combined - scipy.nanmean(combined)) / scipy.nanstd(combined)

        if np.sum(combined.shape) > 0:
            if combined.shape[0] < 9600:
                warnings.warn(f"{s}_{i} Missing a film {combined.shape[0]}")
            for m in valid_films:
                fidx = MOVIES.index(m)
                data = combined[: DURS[fidx]]
                data = np.nan_to_num(data, nan=np.nanmean(data))
                np.savetxt(os.path.join(val, f"Z_sub-{s}_{m}_{i}.csv"), data)
                combined = combined[DURS[fidx] :]

# ## read new z-scored data, combine and calculate correlation with consensus annotation
matches = np.zeros([len(MOVIES), len(its)])
ccc = {}

for m in MOVIES:
    # ## Read in the consensus annotation
    gt = pd.read_csv(
        os.path.join(main, f"Annot_{m}_stim.tsv"), delimiter="\t", header=None
    )
    gt.columns = j_dic["Columns"]
    for it in its:
        files = glob.glob(os.path.join(val, f"Z_sub-*{m}_{it}.csv"))

        combined = []
        for f in files:
            s_val = np.genfromtxt(f)
            if combined == []:
                combined = np.nan_to_num(s_val, nan=np.nanmean(s_val))
            else:
                combined = np.vstack(
                    [combined, np.nan_to_num(s_val, nan=np.nanmean(s_val))]
                )
        combined = combined.T

        combined = combined.squeeze()
        if combined.ndim > 1:
            ave = np.nanmean(combined, axis=1)  # average across subjects
        else:
            ave = combined

        ave = pd.DataFrame(ave)
        new = pd.concat(
            [gt[it], ave], axis=1
        )  # make a df with the ground truth and the validation time course
        co = new.corr()
        matches[MOVIES.index(m), its.index(it)] = np.array(co)[0, 1]

qc = pd.DataFrame(matches, columns=its, index=MOVIES)
match = matches.flatten()

fig = plt.figure(figsize=(15, 6), dpi=300)
# create grid for different subplots
spec = gridspec.GridSpec(ncols=2, nrows=1)

ax0 = fig.add_subplot(spec[0])
bins = np.arange(-1, 1, 0.05)
ax0.hist(match, bins=bins, ec="black", color="darkblue", alpha=0.8)
ax0.set_ylabel("Count")
ax0.set_xlabel("Correlation between Validation and Consensus Annotation")

print(np.nanmean(match))
ax1 = fig.add_subplot(spec[1])
bins = np.arange(0, 1, 0.05)
match = np.mean(matches, axis=0)
ax1.hist(match, bins=bins, alpha=0.8, color="darkblue", ec="black")
ax1.set_ylabel("Count")
ax1.set_xlabel("Mean Correlation between Validation and Consensus Annotation by item")


fig.savefig(os.path.join(outdir, "Corr_VALI.png"))
