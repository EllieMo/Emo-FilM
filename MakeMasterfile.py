#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 09:36:17 2022

@author: elliemorgenroth
"""
# This is directly from the internet to get larger outputs
from IPython.core.display import HTML, display

display(HTML("<style>div.output_scroll { height: 44em; }</style>"))
# Suppress warnings
import warnings

warnings.filterwarnings("ignore")

import glob
import itertools
import json
import math
import os

# Get some useful packages loaded
import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
from pandas import DataFrame, read_csv, read_excel
from scipy import stats

from helper_annot import *

# Set important paths
# adapt USER and paths for your local environment

root = "/Volumes/Sinergia_Emo/Emo-FilM/"
out = f"{root}Annotstudy/derivatives/"
temp = f"/Volumes/Sinergia_Emo/Emo-FilM/temp/"

# Changes directory to your local path
os.chdir(root)

max_zscore = 15  # z-threshold for removal of files for outliers
threshold = 0.20  # threshold for removal of files for agreements

# select participants and movies
# adapt selection of movies and participants if you don't want everything at once
## Excluded bad movies and items for ease, but will have to go back to check this
all_participants = [
    "mode",
    "area",
    "bird",
    "hall",
    "user",
    "oven",
    "army",
    "road",
    "cell",
    "poem",
    "food",
    "town",
    "year",
    "news",
    "goal",
    "week",
    "mall",
    "beer",
    "gate",
    "gene",
    "desk",
    "unit",
    "disk",
    "meat",
    "king",
    "debt",
    "idea",
    "soup",
    "city",
    "girl",
    "dirt",
    "role",
    "poet",
    "song",
    "fact",
    "lake",
    "bath",
    "nice",
    "path",
    "bite",
    "loan",
    "chat",
    "zone",
    "zeal",
]

moves = {
    "AfterTheRain": 496,
    "BetweenViewings": 808,
    "BigBuckBunny": 490,
    "Chatter": 405,
    "FirstBite": 599,
    "LessonLearned": 667,
    "Payload": 1008,
    "Sintel": 722,
    "Spaceman": 805,
    "Superhero": 1028,
    "TearsOfSteel": 588,
    "TheSecretNumber": 784,
    "ToClaireFromSonny": 402,
    "YouAgain": 798,  # ,'RidingTheRails':794,'DamagedKungFu':922
}

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
    "Heartrate",
    "Throat",
    "Stomach",
    "Warm",
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
    "Sad",  # ,'Jaw', 'EyesOpen', 'Breathing', 'Movement','Consequences'
]

## Different Quality Control information
excluded = [0, 0]  # Files excluded for flat or outliers
bad_ann = []  # List of Bad Annotations, so we can follow which ones they are
five = []  # List of Files removed because they are worst of five
numberAnnot = {
    "3": 0,
    "4": 0,
    "_3": 0,
    "_4": 0,
}  # number of annotations making a ground truth '_' marks that one was removed for agreement/five
zfiles = True
saving = False

if zfiles == True:
    for p in all_participants:  # Loop over participants to find files
        for n in its:  # Loop over items
            group = np.array([])
            val_films = []
            for mix, movie in enumerate(moves):  # Loop over films
                files = glob.glob(
                    f"{root}Annotstudy/sub-{p}/beh/sub-{p}_task-{movie}_recording-{n}_stim.tsv.gz"
                )
                if len(files) > 4:
                    print(f"greater 4, {len(files)}")
                for m in files:
                    pre_excluded = sum(excluded)
                    group, excluded = load_data(
                        m, max_zscore, group, excluded
                    )  # Load data and add to group (or not)
                    if pre_excluded < sum(excluded):
                        continue
                    else:
                        val_films.append(movie)
            # z score data
            zgroup = stats.zscore(group)

            for num, val in enumerate(val_films):
                zdata = zgroup[0 : moves[val]]
                if len(zdata) not in moves.values():
                    print("ALERT")

                zgroup = zgroup[moves[val] :]
                np.savetxt(
                    f"{temp}sub-{p}_task-{val}_recording-{n}_stim.tsv",
                    zdata,
                    fmt="%.6f",
                    delimiter="\t",
                )

## Prepare variables for Agreement, Weights and final Time Courses
ccc = {}  # All Agreement scores
mean_ccc = np.ones((len(moves), len(its)))  # Mean Agreement scores

MeanTC = {}  # Final time courses as a dictionary
for mix, movie in enumerate(moves):  # Loop over films
    for iix, n in enumerate(its):  # Loop over items
        group = np.array([])  # Start with an empty array to put the annotators TCs in
        labels = []  # Empty list of who is annotating this pairing
        for six, p in enumerate(
            all_participants
        ):  # Loop over participants to find files
            files = glob.glob(f"{temp}sub-{p}_task-{movie}_recording-{n}_stim.tsv")
            for m in files:
                labels.append(p)  # Append participant labels here
                pre_excluded = excluded  # Check if the file is used or not
                series = read_csv(m, header=None, delimiter="\t", names=["y"])
                if np.shape(group)[0] == 0:
                    group = series
                else:
                    group = np.hstack([group, series])
        if group.shape[1] > 2:
            ## All things agreement start here as group is completed for this pairing
            # First calculate all cccs for this filmxitem combination
            for i, j in enumerate(itertools.combinations(range(group.shape[1]), 2)):
                ccc[movie + "_" + n + "_" + str(j[0]) + "_" + str(j[1])] = lins_ccc(
                    group[:, j[0]], group[:, j[1]]
                )

            # Calculate the mean_ccc for this filmxitem combination
            mean_ccc[mix, iix] = np.mean(
                [ccc[z] for z in ccc.keys() if z.find(movie + "_" + n) == 0]
            )

            ## Find out if leaving one out will improve agreement
            ccc_loolist = []  ## List of CCCs if one annotator is left out
            ccc_loo = {}  ## Mean CCCs if one annotator is left out
            for q in range(group.shape[1]):
                ccc_loolist = [
                    ccc[z]
                    for z in ccc.keys()
                    if z.find(str(q)) == -1 and z.find(movie + "_" + n) >= 0
                ]
                ccc_loo[q] = np.mean(ccc_loolist)

            best_ccc = max(
                ccc_loo.values()
            )  # Best CCC possible after leaving out a participant
            wr_idx = [i for i in ccc_loo if ccc_loo[i] == best_ccc][
                0
            ]  # Worst raters index in this filmxitem combination
            worst_rater = labels[wr_idx]  # Worst raters label (to find index overall)
        elif group.shape[1] <= 2:
            print(f"ALERT, only {str(group.shape[1])} raters left {movie}_{n}")
            print(
                "ALERT, this case should NOT happen when all films and items are selected"
            )
            mean_ccc[mix, iix] = lins_ccc(group[:, 0], group[:, 1])

        if group.shape[1] <= 3:  # If there are only 3 or less annotators
            if (
                best_ccc - mean_ccc[mix, iix]
            ) >= threshold:  # Check that there isn't a major outlier in these
                print(f"ALERT, only {str(group.shape[1])} raters left {movie}_{n}")
                print(
                    f"Agreement is {str(mean_ccc[mix,iix])} instead of {str(best_ccc)}"
                )
                print(np.shape(group.shape))
            numberAnnot["3"] += 1

        elif group.shape[1] == 4:  # Standard case is if there are 4 annotators
            if (
                best_ccc - mean_ccc[mix, iix]
            ) >= threshold:  # Check if exclusion would make the mean better
                # Erase worst rater from everything if True
                for i in [
                    z
                    for z in ccc.keys()
                    if z.find(str(wr_idx)) != -1 and z.find(movie + "_" + n) == 0
                ]:
                    del ccc[i]

                bad_ann.append(f"{movie}_{worst_rater}_{n}")
                group = np.delete(group, wr_idx, 1)
                mean_ccc[mix, iix] = best_ccc
                numberAnnot["_3"] += 1
            else:
                numberAnnot["4"] += 1
        elif (
            group.shape[1] == 5
        ):  # In the rare case of 5 annotators we remove the worst in any case
            print(f"Group size is 5 for item {n} and film {movie}")
            # Erase worst rater from everything
            for i in [
                z
                for z in ccc.keys()
                if z.find(str(wr_idx)) != -1 and z.find(movie + "_" + n) == 0
            ]:
                del ccc[i]
            five.append(f"{movie}_{worst_rater}_{n}")
            group = np.delete(group, wr_idx, 1)
            mean_ccc[mix, iix] = best_ccc
            numberAnnot["_4"] += 1
        elif (
            group.shape[1] > 5
        ):  # In the rare case of 5 annotators we remove the worst in any case
            print(f"Group size is 6 for item {n} and film {movie}")
            # Erase worst rater from everything
            for i in [
                z
                for z in ccc.keys()
                if z.find(str(wr_idx)) != -1 and z.find(movie + "_" + n) == 0
            ]:
                del ccc[i]
            for i in [
                z
                for z in ccc.keys()
                if z.find(str(wr_idx)) != -1 and z.find(movie + "_" + n) == 0
            ]:
                del ccc[i]
            five.append(f"{movie}_{worst_rater}_{n}")
            five.append(f"{movie}_second_{n}")
            group = np.delete(group, wr_idx, 1)
            mean_ccc[mix, iix] = best_ccc
            numberAnnot["_4"] += 1

        ## Add final group mean to MeanTC, this is the ground truth
        MeanTC[movie + "_" + n] = np.mean(group, axis=1)

## Print QC information
print()
print(f"{str(sum(excluded))} annotations excluded")
print(f"{str(excluded[0])} annotations removed for flat")
print(f"{str(excluded[1])} annotations removed for outliers")
print(f"{len(bad_ann)} annotations removed for agreement")
print(f"{len(five)} annotations removed as they were worst of 5")
print(numberAnnot)

print("Mean of completed CCC values:")
print(np.nanmean(list(ccc.values())))

ccct = np.array(list(ccc.values()))
np.save(f"{out}ccc_values", ccct)
cccixm_df = DataFrame(mean_ccc, index=moves, columns=its)
np.save(f"{out}mean_ccc", cccixm_df)

durs = []
AMTC = []
mfilm = np.zeros([len(its), len(moves)])

for iix, n in enumerate(its):  # Loop over items
    MTC = []
    for mix, movie in enumerate(moves):  # Loop over films
        if iix == 0:
            durs.append(np.shape(MeanTC[movie + "_" + n])[0])
        TC = MeanTC[movie + "_" + n]
        mfilm[iix, mix] = np.mean(MeanTC[movie + "_" + n])
        try:
            MTC = np.hstack([MTC, TC])
        except:
            MTC = TC
    MTC = MTC - np.mean(MTC)
    try:
        AMTC = np.vstack([AMTC, MTC])
    except:
        AMTC = MTC

fig = plt.figure(figsize=(15, 5), dpi=300)
sn.heatmap(
    mfilm.T, square=True, xticklabels=its, yticklabels=moves, cmap="coolwarm", center=0
)
fig.savefig(
    "/Volumes/Sinergia_Emo/EPFL_drive/Sinergia Project/Writing/Data_Paper/Figures/Figure2.png",
    bbox_inches="tight",
)

print(f"mean df = {np.mean(np.mean(AMTC))}")
print(f"var df = {np.mean(np.var(AMTC))}")
print(f"max df = {np.mean(np.max(AMTC))}")
print(f"min df = {np.mean(np.min(AMTC))}")

if saving == True:
    saveTC = DataFrame(AMTC.transpose())
    saveTC.to_csv(f"{out}C_Annot_FILMS_stim.tsv", sep="\t", header=False, index=False)

    for mix, movie in enumerate(moves):
        file = AMTC[:, : moves[movie]]
        AMTC = AMTC[:, moves[movie] :]

        df = DataFrame(file.transpose())
        df.to_csv(f"{out}Annot_{movie}_stim.tsv", sep="\t", header=False, index=False)
