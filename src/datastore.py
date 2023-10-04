#!/usr/bin/env python3
import logging
import logging.handlers
import sys
import os

from rdflib import Graph, Literal, BNode
from rdflib.collection import Collection
from rdflib.namespace import Namespace, XSD, RDF

lenti_ns = Namespace("http://example.org/lenti#")

# Logging
logger = logging.getLogger("datastore")
logger.setLevel(logging.DEBUG)

filehandler = logging.FileHandler("../log/datastore.log")
consolehandler = logging.StreamHandler(sys.stdout)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
filehandler.setFormatter(formatter)
consolehandler.setFormatter(formatter)

logger.addHandler(filehandler)


class SavedModel():

    def __init__(self, taskLabel, profilesTrainedOn, measureCollection):
        # doSomething
        self.taskLabel = taskLabel
        self.modelTotalItems = profilesTrainedOn
        self.measureCollection = measureCollection

    def getTaskLabel(self):
        return self.taskLabel

    def getModelTotalItems(self):
        return self.modelTotalItems

    def getMeasureCollection(self):
        return self.measureCollection


class QualityMeasure:

    def __init__(self, measureLabel, semanticDescription, mean, median, medianEps):
        self.measureLabel = measureLabel
        self.semDesc = semanticDescription
        self.mean = mean
        self.median = median
        self.medianEps = medianEps

    def getMeasureLabel(self):
        return self.measureLabel

    def getSemanticDescription(self):
        return self.semDesc

    def getMeanValue(self):
        return self.mean

    def getMedianValue(self):
        return self.median

    def getMedianEpsilonValue(self):
        return self.medianEps


class Datastore:
    """
        Datastore class for Lenti
    """

    class __Datastore:
        def __init__(self):
            self.datastore = Graph(identifier='lenti_graph')
            logger.debug("Loading datastore")
            if (not(os.path.isfile('../datastore/lentistore.nt'))):
                logger.debug("Creating datastore")
                open('../datastore/lentistore.nt', 'w').close()

            self.datastore.parse('../datastore/lentistore.nt', format="nt")

    instance = None

    def __init__(self):
        if not Datastore.instance:
            Datastore.instance = Datastore.__Datastore()

        self.datastore = Datastore.instance.datastore

    def getTrainedModelURI(self, label):
        logger.debug("Checking datastore for %s trained model", str(label))
        getModelURI = self.datastore.triples((None, lenti_ns.title, Literal(label, datatype=XSD.string)))
        subject = None
        for (s, p, o) in getModelURI:
            subject = s

        if (subject is None):
            logger.debug("No trained model for %s", str(label))
            return None
        else:
            return subject

    def getTrainedModel(self, label):
        # Query the KG to find a trained model with the same label
        # return None if the model does not exist or a Model object

        logger.debug("Checking datastore for %s trained model", str(label))
        getModelURI = self.datastore.triples((None, lenti_ns.title, Literal(label, datatype=XSD.string)))
        subject = None
        for (s, p, o) in getModelURI:
            subject = s

        if (subject is None):
            logger.debug("No trained model for %s", str(label))
            return None
        else:
            logger.debug("Loading model for %s", str(label))
            profilesTrainedOnTriples = self.datastore.triples((subject, lenti_ns.profilesTrainedOn, None)) # integer
            (s, p, o) = profilesTrainedOnTriples.__next__()
            profilesTrainedOn = o.toPython()

            hasMeasureCollectionTriples = self.datastore.triples((subject, lenti_ns.hasMeasureCollection, None)) # uri
            (s, p, o) = hasMeasureCollectionTriples.__next__()
            hasMeasureCollection = Collection(self.datastore, o)
            measureCollection = []
            for i in range(0, hasMeasureCollection.__len__()):
                measureURI = hasMeasureCollection[i]
                (s, p, o) = self.datastore.triples((measureURI, lenti_ns.label, None)).__next__()
                measureLabel = o.toPython()
                (s, p, o) = self.datastore.triples((measureURI, lenti_ns.semanticDescription, None)).__next__()
                semDesc = o.toPython()
                (s, p, o) = self.datastore.triples((measureURI, lenti_ns.mean, None)).__next__()
                mean = o.toPython()
                (s, p, o) = self.datastore.triples((measureURI, lenti_ns.median, None)).__next__()
                median = o.toPython()
                (s, p, o) = self.datastore.triples((measureURI, lenti_ns.medianEpsilon, None)).__next__()
                medianEps = o.toPython()

                measure = QualityMeasure(measureLabel, semDesc, mean, median, medianEps)
                measureCollection.append(measure)

            loadedModel = SavedModel(label, profilesTrainedOn, measureCollection)
            return loadedModel

    def createTrainedModel(self, model, task_description):
        # Have to think about locks for concurrent access or just store everything at once
        # if there is another model with same label, then return false and do not store anything
        logger.debug("Creating triples for trained model in local knowledge graph")

        getModelURI = BNode()
        self.datastore.add((getModelURI, RDF.type, lenti_ns.TrainedTask))
        self.datastore.add((getModelURI, lenti_ns.title, Literal(model.getTaskLabel(), datatype=XSD.string)))
        self.datastore.add((getModelURI, lenti_ns.description, Literal(task_description, datatype=XSD.string)))
        self.datastore.add((getModelURI, lenti_ns.profilesTrainedOn, Literal(model.getModelTotalItems(), datatype=XSD.integer)))

        measureCollectionURI = BNode()
        self.datastore.add((measureCollectionURI, RDF.type, lenti_ns.MeasuresCollection))

        measureCollection = Collection(self.datastore, measureCollectionURI)
        for m in model.getMeasureCollection():
            measureURI = BNode()
            self.datastore.add((measureURI, RDF.type, lenti_ns.Measure))
            self.datastore.add((measureURI, lenti_ns.label, Literal(m.getMeasureLabel(), datatype=XSD.string)))
            self.datastore.add((measureURI, lenti_ns.semanticDescription, Literal(m.getSemanticDescription(), datatype=XSD.string)))
            self.datastore.add((measureURI, lenti_ns.mean, Literal(m.getMeanValue(), datatype=XSD.double)))
            self.datastore.add((measureURI, lenti_ns.median, Literal(m.getMedianValue(), datatype=XSD.double)))
            self.datastore.add((measureURI, lenti_ns.medianEpsilon, Literal(m.getMedianEpsilonValue(), datatype=XSD.double)))

            measureCollection.append(measureURI)

        self.datastore.add((getModelURI, lenti_ns.hasMeasureCollection, measureCollectionURI))

        logger.debug("Local Knowledge Graph updated")
        self.datastore.serialize('../datastore/lentistore.nt', format="nt")

    def updateKnowledgeGraph(self, model, task_description):
        logger.debug("Updating local Knowledge Graph")
        modelURI = self.getTrainedModelURI(model.getTaskLabel())
        if modelURI:
            hasMeasureCollectionTriples = self.datastore.triples((modelURI, lenti_ns.hasMeasureCollection, None)) # uri
            (s, p, o) = hasMeasureCollectionTriples.__next__()
            hasMeasureCollection = Collection(self.datastore, o)
            for i in range(0, hasMeasureCollection.__len__()):
                measureURI = hasMeasureCollection[i]
                self.datastore.remove((measureURI, None, None))

            hasMeasureCollection.clear()
            self.datastore.remove((o, None, None))
            self.datastore.remove((modelURI, None, None))
            self.createTrainedModel(model, task_description)
