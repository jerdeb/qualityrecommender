#!/usr/bin/env python3

import cherrypy
import logging
import logging.handlers
import sys
import json
import uuid

from model import Model
import pandas as pd
from os import listdir
import os

from datastore import Datastore

# Logging
logger = logging.getLogger("ws")
logger.setLevel(logging.DEBUG)

filehandler = logging.FileHandler("../log/ws.log")
consolehandler = logging.StreamHandler(sys.stdout)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
filehandler.setFormatter(formatter)
consolehandler.setFormatter(formatter)

logger.addHandler(filehandler)


class WebService(object):

    def outputRecommendationModel(self, recommendedModel):
        for key, value in recommendedModel.items():
            print(key, self._prettyPrint(value))

    def _prettyPrint(self, val):
        return str(round((val*100.0), 2))+"%"

    def startup(self):
        print("Starting up Lenti")

        logger.info("Loading Datastore")
        self.datastore = Datastore()
        self.trainingModels = {}
        self.dumpstore = {}

        configDF = pd.read_csv("""../training/config.lenti""")
        for index, row in configDF.iterrows():
            label = row['task_name']
            trainedModel = self.datastore.getTrainedModel(label)
            self.dumpstore[label] = row['datastore_folder']
            if not (os.path.isdir("../datastore/"+row['datastore_folder'])):
                os.mkdir("../datastore/"+row['datastore_folder'])

            if (not trainedModel):
                print("Training Models")
                df = pd.read_csv("""../training/"""+row['training_data_name'])
                training = Model(df, taskLabel=label)

                # Update the model with the user selection quality profiles in case of reboot
                if (os.path.isdir("../datastore/"+row['datastore_folder'])):
                    for f in listdir("../datastore/"+row['datastore_folder']):
                        filename, file_extension = os.path.splitext(f)
                        if (file_extension == ".json"):
                            qualityProfile = dict({})
                            with open("../datastore/"+row['datastore_folder']+"/"+f) as the_file:
                                data = json.load(the_file)
                                for key, value in data.items():
                                    qualityProfile[key] = value

                                training.updateModel(qualityProfile)
            else:
                training = Model(taskLabel=label, savedModel=trainedModel)

            self.trainingModels[label] = training

        print("Finished Training Models")

    # @cherrypy.expose
    # @cherrypy.tools.json_out()
    # def get_recommendation(self):
    #     logger.debug("Returning Recommendation")
    #     recModel = self.trainingModels["Library Metadata Interlinking"].recommendQualityMeasures(threshold=0.01)
    #     return json.dumps(recModel)

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def update_measures(self):
        logger.debug("Updating Measures Request")
        data = cherrypy.request.json
        logger.debug(data)

        with open("../datastore/"+self.dumpstore["Library Metadata Interlinking"]+"/"+str(uuid.uuid4().hex)+'.json','w') as outfile:
            json.dump(data, outfile)

        qualityProfile = dict({})
        for key, value in data.items():
            qualityProfile[key] = value

        return json.dumps(self.trainingModels["Library Metadata Interlinking"].updateModel(qualityProfile))

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def get_recommendation(self):
        logger.debug("Returning Recommendation")
        data = cherrypy.request.json
        logger.debug(data)

        task = str(data["Task"])
        chosenthreshold = float(data["Threshold"])

        recModel = self.trainingModels[task].recommendQualityMeasures(threshold=chosenthreshold)
        return json.dumps(recModel)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def get_trainedmodels(self):
        logger.debug("Getting Trained Models")
        trainedModels = {}
        trainedModels["Models"] = list(self.trainingModels.keys())
        return json.dumps(trainedModels)


if __name__ == '__main__':
    ws = WebService()
    ws.startup()
    cherrypy.config.update({
                            'server.socket_host': '0.0.0.0',
                            'server.socket_port': 8081,
                          })
    cherrypy.quickstart(ws, "/recommender")
