#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 15:15:36 2023

@author: elliemorgenroth

This makes a lot of plots visualizing the inter-rater correlations from the annot study
"""
import os
import sys

import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import gridspec
from matplotlib import pyplot as plt

from constants_emo_film import ITEM_CATS, ITS, MOVES


# Fix paths according to where everything is and where you want to save things
root = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/Sinergia_Emo/Emo-FilM/"
savedir = sys.argv[2] if len(sys.argv) > 2 else "/Volumes/Sinergia_Emo/EPFL_drive/Sinergia Project/Writing/Data_Paper/Figures/"

ccc = np.load(os.path.join(root, "derivatives", "ccc_values.npy"), allow_pickle=True)

# # Plot agreement
plt.figure()
bins = np.arange(-1, 1, 0.05)
plt.hist(ccc, bins=bins, alpha=0.5, ec="black")
plt.title("Distribution of complete mean correlation")
plt.xlabel("Correlation")
plt.ylabel("count")

cccixm_df = np.load(os.path.join(root, "derivatives", "mean_ccc.npy"), allow_pickle=True)
cccixm_df = pd.DataFrame(cccixm_df, index=MOVES, columns=ITS)

# Sorting doesn't work here, Mean ends up in the middle ...
# @Ellie What are you trying to do? sort_values sorts the entries by their mean values.
# Are you trying to move the "Mean" column to a certain point in the dictionary?
# And are you trying to average by column and by row?  
cccixm_df["Mean"] = np.mean(cccixm_df, axis=1)
cccixm_df = cccixm_df.T
cccixm_df["Mean"] = np.mean(cccixm_df, axis=1)
cccixm_df["Mean"]["Mean"] = None
cccixm_df = cccixm_df.sort_values(by=["Mean"], ascending=False)
cccixm_df = cccixm_df.T
cccixm_sort = cccixm_df.sort_values(by=["Mean"], ascending=False)
cccixm_sort = cccixm_sort.T

agreements = cccixm_sort.T
file = pd.DataFrame()
group_mean = pd.DataFrame()
colours = [
    "lightblue",
    "darkturquoise",
    "aquamarine",
    "salmon",
    "sandybrown",
    "khaki",
    "indianred",
]

# @Stef check this for loop with data
c_list = []
for i in ITEM_CATS:
    grow = agreements[ITEM_CATS[i]]
    grow = grow.drop("Mean")
    grow = grow.reindex(grow.mean().sort_values(ascending=False).index, axis=1)
    try:
        file = pd.concat([file, grow], axis=1)
        group_mean = np.vstack([group_mean, grow.mean(axis=1).to_numpy().flatten()])
    except:
        file = grow
        group_mean = grow.mean(axis=1).to_numpy().flatten()
    for j in ITEM_CATS[i]:
        c_list.append(colours[list(ITEM_CATS.keys()).index(i)])

ilist = file.columns
group_mean = pd.DataFrame(group_mean.T, columns=list(ITEM_CATS.keys()))

##############################################################################

# Calculate the histogram
counts, bns, patches = plt.hist(
    ccc, bins=bins, color="darkblue", alpha=0.8, edgecolor="black"
)

# Reverse the counts array to make the bars increase towards the left
counts = counts[::-1]

# Reverse the bins array to match the reversed counts
bns = bns[::-1]

# Calculate the widths of the bars (negative values)
widths = -np.diff(bns)

#############################################################################
# create a figure
fig = plt.figure(figsize=(15, 5), dpi=300)

# create grid for different subplots
spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[1, 3])
# initializing x,y axis value
bins = np.arange(-0.6, 1, 0.05)

# ax0 will take 0th position in
# geometry(Grid we created for subplots),
# as we defined the position as "spec[0]"

ax0 = fig.add_subplot(spec[0])
ax0.grid(axis="y", zorder=0)

# Create the horizontal bars
ax0.barh(
    bns[:-1],
    counts,
    height=widths,
    align="center",
    color="darkblue",
    alpha=0.8,
    edgecolor="black",
)

# Invert the x-axis
plt.gca().invert_xaxis()

# Put the y-axis on the right
plt.gca().yaxis.tick_right()

# Set the limITS and labels
ax0.set_ylim(-0.6, 1)
ax0.set_ylabel("Correlation between raters")
ax0.set_xlabel("Count")

# ax1 will take 0th position in
# geometry(Grid we created for subplots),
# as we defined the position as "spec[1]"
ax1 = fig.add_subplot(spec[1], sharey=ax0)
ax1.grid(axis="y", zorder=0)
ax1.boxplot(
    file.T,
    patch_artist=True,
    labels=file.index,
    showmeans=True,
    boxprops=dict(alpha=0.8, facecolor="darkblue", color="darkblue"),
    meanprops={"markerfacecolor": "white", "markeredgecolor": "white"},
    medianprops={"color": "white"},
)
ax1.set_ylim(ax0.get_ylim())

ax1.set_xticklabels(file.index, rotation=90)

# display the plots
plt.show()
fig.savefig(os.path.join(savedir, "AnnotQC_1.png"), bbox_inches="tight")

##############################################################################
# create a figure
fig = plt.figure(figsize=(15, 5), dpi=300)

# create grid for different subplots
spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[4, 2])

# initializing x,y axis value
bins = np.arange(-0.2, 0.9, 0.05)

ax0 = fig.add_subplot(spec[0])
boxplot = ax0.boxplot(
    file,
    patch_artist=True,
    labels=ilist,
    showmeans=True,
    meanprops={"markerfacecolor": "black", "markeredgecolor": "black"},
    medianprops={"color": "black"},
)
ax0.set_xticklabels(ilist, rotation=90)
ax0.set_ylabel("Correlation between raters")
ax0.grid(axis="y", zorder=0)
for patch, color in zip(boxplot["boxes"], c_list):
    patch.set_facecolor(color)
plt.gca().yaxis.tick_right()
# Put the y-axis on the right

ax1 = fig.add_subplot(spec[1], sharey=ax0)
ax1.grid(axis="y", zorder=0)
ax1.set_xticklabels(ilist, rotation=90)

sn.violinplot(ax=ax1, data=group_mean, palette=colours[0:6])

plt.show()
fig.savefig(os.path.join(savedir, "AnnotQC_2.png"), bbox_inches="tight")
##############################################################################
