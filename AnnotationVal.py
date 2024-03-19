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

# Get some useful packages loaded
import numpy as np
import pandas as pd
import scipy
from mat4py import loadmat
from matplotlib import gridspec
from matplotlib import pyplot as plt
from pandas import read_csv, read_excel

# Paths to where everything is and where results are saved
source = "/Volumes/Sinergia_Emo/Emo-FilM/"
save = f"{source}fMRIstudy/"

main = f"{source}Annotstudy/derivatives/"
val = f"{source}fMRIstudy/derivatives/"

movs = [
    "AfterTheRain",
    "BetweenViewings",
    "BigBuckBunny",
    "Chatter",
    "FirstBite",
    "LessonLearned",
    "Payload",
    "Sintel",
    "Spaceman",
    "Superhero",
    "TearsOfSteel",
    "TheSecretNumber",
    "ToClaireFromSonny",
    "YouAgain",
]
durs = [496, 808, 490, 405, 599, 667, 1008, 722, 805, 1028, 588, 784, 402, 798]

j_file = open(f"{main}Annot_AfterTheRain_stim.json")
j_dic = json.load(j_file)

emo = {}
emo["Com"] = j_dic["Columns"][:37]
emo["E13"] = j_dic["Columns"][37:]

# This splits the items in the six groups.
item_cats = {}
item_cats["appraisal"] = j_dic["Columns"][0:10]
item_cats["expression"] = j_dic["Columns"][10:15]
item_cats["motivation"] = j_dic["Columns"][15:25]
item_cats["feeling"] = j_dic["Columns"][25:32]
item_cats["physiology"] = j_dic["Columns"][32:37]
item_cats["emotion"] = j_dic["Columns"][37:50]

# items not used in validation: Heartrate, Warm

subs = [
    "S01",
    "S02",
    "S03",
    "S04",
    "S05",
    "S06",
    "S07",
    "S08",
    "S09",
    "S10",
    "S11",
    "S13",
    "S14",
    "S15",
    "S16",
    "S17",
    "S19",
    "S20",
    "S21",
    "S22",
    "S23",
    "S24",
    "S25",
    "S26",
    "S27",
    "S28",
    "S29",
    "S30",
    "S31",
    "S32",
]
sess = ["1", "2", "3", "4"]  # '5'

val_times = loadmat(val + "validation/ValTimes.mat")

val_its = loadmat(val + "validation/ValItems.mat")
val_its = np.asarray(list(val_its.values())).flatten().reshape(5, 32).transpose()
# data = np.array(data[12]).flatten().reshape(-1,2)[:,1]

f = "/Volumes/Sinergia_Emo/EPFL_drive/Sinergia Project/Writing/Data_Paper/Supplementary Tables.xlsx"
meta_data = read_excel(f, header=1)

for i in subs:
    sidx = int(i[1:])
    items = list(val_its[sidx - 1])

    # ## Lists files as returned from our acquisition
    files = glob.glob(f"{save}sub-{i}/ses*/beh/sub-{i}_*_task-*_events.tsv")

    # ## Loop Over list of all _val files
    for file in files:
        movie = file.split("_")[-2].split("-")[1]
        movie_idx = movs.index(movie)

        # ## Get time stamps for annotated clips
        vTimes = np.asarray(list(val_times[movie]))

        vali = pd.read_csv(file, delimiter="\t")

        # ## Load validation file

        vali = np.asarray(vali)
        vali = vali[:, 2:7]

        # ## Arrange files to the full length for each item
        for m in range(np.shape(vali)[1]):
            item = vali[:, m]
            itemH = items[m]
            n_times = np.zeros(durs[movie_idx])
            n_times[:] = np.nan
            for l in range(len(vTimes)):
                tim = vTimes[l].tolist()
                try:
                    n_times[tim[0]: tim[1]] = item[l]
                except:
                    continue
                np.savetxt(
                    val + "validation/sub-" + i + "_" + movie + "_" + itemH + ".csv",
                    n_times,
                )


its = [
    "Standards",
    "PleasantSelf",
    "SocialNorms",
    "PleasantOther",
    "GoalsOther",
    "Controlled",
    "Predictable",
    "Suddenly",
    "Agent",
    "Urgency",
    "Lips",
    "Tears",
    "Eyebrows",
    "Smile",
    "Frown",
    "Stop",
    "Undo",
    "Repeat",
    "Oppose",
    "Attention",
    "Tackle",
    "Command",
    "Support",
    "Move",
    "Care",
    "Bad",
    "Good",
    "Calm",
    "Strong",
    "IntenseEmotion",
    "Alert",
    "AtEase",
    "Muscle",
    "Throat",
    "Stomach",
    "Anger",
    "Guilt",
    "WarmHeartedness",
    "Disgust",
    "Happiness",
    "Fear",
    "Regard",
    "Anxiety",
    "Satisfaction",
    "Pride",
    "Surprise",
    "Love",
    "Sad",
]

# ## makes continuous time course, z-score within subject and saves data
for i in sorted(its):
    for s in subs:
        files = glob.glob(f"{val}validation/sub-{s}_*_{i}.csv")

        combined = []
        valid_films = []
        for f in sorted(files):
            valid_films.append(f.split("_")[-2])
            # s_val = np.genfromtxt(f)
            # s_val = pd.DataFrame(s_val)
            s_val = pd.read_csv(f, header=None)
            s_val = s_val.interpolate(method="linear")  # , order = 1)
            s_val = s_val.to_numpy()

            try:
                combined = np.vstack([combined, s_val])
            except:
                combined = s_val

        # ## z-score
        combined = (combined - scipy.nanmean(combined)) / scipy.nanstd(combined)

        if np.sum(combined.shape) > 0:
            if combined.shape[0] < 9600:
                print(f"{s}_{i} Missing a film {combined.shape[0]}")
            for m in valid_films:
                fidx = movs.index(m)
                data = combined[: durs[fidx]]
                data = np.nan_to_num(data, nan=np.mean(data))
                np.savetxt(f"{val}validation/Z_sub-{s}_{m}_{i}.csv", data)
                combined = combined[durs[fidx] :]

# ## read new z-scored data, combine and calculate correlation with consensus annotation
matches = np.zeros([len(movs), len(its)])
ccc = {}
for m in movs:
    # ## Read in the consensus annotation
    gt = read_csv(f"{main}Annot_{m}_stim.tsv.gz", delimiter="\t", header=None)
    gt.columns = j_dic["Columns"]
    for it in its:
        files = glob.glob(f"{val}validation/Z_sub-*{m}_{it}.csv")

        combined = []
        for f in files:
            s_val = np.genfromtxt(f)
            s_val = np.nan_to_num(s_val, nan=np.nanmean(s_val))
            try:
                combined = np.vstack([combined, s_val])
            except:
                combined = s_val
        combined = combined.T

        try:
            ave = np.mean(combined, axis=1)  # average across subjects
        except:
            ave = combined
        ave = pd.DataFrame(ave)
        new = pd.concat(
            [gt[it], ave], axis=1
        )  # make a df with the ground truth and the validation time course
        co = new.corr()
        matches[movs.index(m), its.index(it)] = np.array(co)[0, 1]

# cc = pd.DataFrame(ccc)
qc = pd.DataFrame(matches, columns=its, index=movs)
match = matches.flatten()

fig = plt.figure(figsize=(15, 6), dpi=300)
# create grid for different subplots
spec = gridspec.GridSpec(ncols=2, nrows=1)

ax0 = fig.add_subplot(spec[0])
bins = np.arange(-1, 1, 0.05)
ax0.hist(match, bins=bins, ec="black", color="darkblue", alpha=0.8)
ax0.set_ylabel("Count")
ax0.set_xlabel("Correlation between Validation and Consensus Annotation")
# ax0.title('Correlation between Validation and Ground truth for each ItemxMovie')

print(np.nanmean(match))
ax1 = fig.add_subplot(spec[1])
bins = np.arange(0, 1, 0.05)
match = np.mean(matches, axis=0)
ax1.hist(match, bins=bins, alpha=0.8, color="darkblue", ec="black")
ax1.set_ylabel("Count")
ax1.set_xlabel("Mean Correlation between Validation and Consensus Annotation by item")
# ax1.title('Mean Correlation between Valdidation and Ground Truth by Item')


fig.savefig(
    "/Volumes/Sinergia_Emo/EPFL_drive/Sinergia Project/Writing/Data_Paper/Figures/Corr_VALI"
)
