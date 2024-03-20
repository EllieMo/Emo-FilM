#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from copy import deepcopy

ALL_PARTICIPANTS_ANNOTATION = [
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

ALL_SUBJECTS_FMRI = [
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

ALL_SESSIONS_FMRI = ["1", "2", "3", "4", "5"]

ALL_MOVIES = {
    "AfterTheRain": 496,
    "BetweenViewings": 808,
    "BigBuckBunny": 490,
    "Chatter": 405,
    "DamagedKungFu": 922,
    "FirstBite": 599,
    "LessonLearned": 667,
    "Payload": 1008,
    "RidingTheRails": 794,
    "Sintel": 722,
    "Spaceman": 805,
    "Superhero": 1028,
    "TearsOfSteel": 588,
    "TheSecretNumber": 784,
    "ToClaireFromSonny": 402,
    "YouAgain": 798,
}

ALL_ITEMS = [
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
    "Regard",
    "Anxiety",
    "Satisfaction",
    "Pride",
    "Surprise",
    "Love",
    "Sad",
]

BAD_ITEMS = ["Jaw", "EyesOpen", "Breathing", "Movement", "Consequences"]
ITS = [a for a in ALL_ITEMS if a not in BAD_ITEMS]

BAD_MOVIES = ["DamagedKungFu", "RidingTheRails"]
MOVIES_DICT = deepcopy(ALL_MOVIES)

for a in BAD_MOVIES:
    MOVIES_DICT.pop(a, None)

MOVIES = list(MOVIES_DICT.keys())
DURS = list(MOVIES_DICT.values())

ITEM_CATS = {}
ITEM_CATS["appraisal"] = ITS[0:10]
ITEM_CATS["expression"] = ITS[10:15]
ITEM_CATS["motivation"] = ITS[15:25]
ITEM_CATS["feeling"] = ITS[25:32]
ITEM_CATS["physiology"] = ITS[32:37]
ITEM_CATS["emotion"] = ITS[37:50]

# items not used in validation: Heartrate, Warm
ITS_VALIDATION = [a for a in ITS if a not in ["Heartrate", "Warm"]]
