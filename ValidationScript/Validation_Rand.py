#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 13:53:23 2021

@author: elliemorgenroth

Randomized allocation of annotation items to participants for movie fMRI project
"""

import random

import numpy as np

# All items from the study
all_items = [
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
    "Consequences",
    "Lips",
    "Tears",
    "Jaw",
    "Eyebrows",
    "Smile",
    "Frown",
    "EyesOpen",
    "Movement",
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
    "Breathing",
    "Throat",
    "Stomach",
    "Warm",
    "Anger",
    "Guilt",
    "WarmHeartedness",
    "Disgust",
    "Happiness",
    "Fear",
    "Contempt",
    "Anxiety",
    "Satisfaction",
    "Shame",
    "Surprise",
    "Love",
    "Sad",
]
# Items we don't want to use for the validation, because of poor quality or replication wiht physio
bad_items = [
    "Jaw",
    "EyesOpen",
    "Breathing",
    "Warm",
    "Consequences",
    "Movement",
    "Heartrate",
]

sel_items = list(set(all_items) - set(bad_items))

# Number of participants here
parts = list(range(30))

# Number of validations per item per participant
num_vals = 3


ran_list = sel_items * num_vals
random.shuffle(ran_list)

its_part = len(ran_list) / len(parts)

for i in parts:
    p_list = []
    for j in range(round(its_part)):
        try:
            while ran_list[0] in p_list:
                random.shuffle(ran_list)

            p_list.append(ran_list[0])
            ran_list.remove(ran_list[0])
        except IndexError:
            break

    while len(p_list) < 5:
        p_list.append("")
    if i == 0:
        ran_val = []
        ran_val = np.asarray(p_list)
    else:
        ran_val = np.row_stack((ran_val, np.asarray(p_list)))

# Save this info - this doesn't currently work, doesn't really matter
# np.savetxt('Val_Items.csv', ran_val, delimiter=',')
