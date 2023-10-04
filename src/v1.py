#!/usr/bin/env python3

# Python3

# This file will contain the implementation as described in the draft
# as the proof of concept

# Imports
import logging
import logging.handlers
import sys
from model import Model


import pandas as pd


# Logging
logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)

filehandler = logging.FileHandler("main.log")
consolehandler = logging.StreamHandler(sys.stdout)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
filehandler.setFormatter(formatter)
consolehandler.setFormatter(formatter)

logger.addHandler(consolehandler)

# Methods


# Training
def train(task, dataframe):
    """
        Trains the algorithm for a particular task using a number
        of quality profiles that were manually defined by users.
    """
    return None

# propagating


# Classes
class Commons():
    def outputRecommendationModel(self, recommendedModel):
        for key, value in recommendedModel.items():
            print(key, self.__prettyPrint(value))

    def __prettyPrint(self, val):
        return str(round((val*100.0), 2))+"%"


# main
df = pd.read_csv("""/Users/jeremy/Documents/Fellowship/Digital Library Survey/Data Quality Profiles/data.csv""")
training = Model(df, taskLabel="library")
recModel = training.recommendQualityMeasures()

print("Recommendation after training")
Commons().outputRecommendationModel(recModel)

print("Updating training model")
import json
qualityProfile = dict({})
with open('/Users/jeremy/Documents/Codebase/Repository/QualityRecommender/tests/p1.json') as f:
    data = json.load(f)
    for key,value in data.items():
        qualityProfile[key] = value

    updated = training.updateModel(qualityProfile)

# for i in range(1,10000):
    # updated = training.updateModel(qualityProfile)

recModel = training.recommendQualityMeasures()
Commons().outputRecommendationModel(recModel)
