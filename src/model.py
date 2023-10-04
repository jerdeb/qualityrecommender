#!/usr/bin/env python3
import logging
import logging.handlers
import sys

import math
import numpy as np

from scipy import stats
from scipy.stats import norm

from datastore import SavedModel, QualityMeasure, Datastore

# Logging
logger = logging.getLogger("model")
logger.setLevel(logging.DEBUG)

filehandler = logging.FileHandler("../log/model.log")
consolehandler = logging.StreamHandler(sys.stdout)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
filehandler.setFormatter(formatter)
consolehandler.setFormatter(formatter)

logger.addHandler(filehandler)


class _Median():
    """
        Private helper class to calculate incremental median.
        Implements the moving median as describe in
        Feldman and Shavitt FAME.

        see: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.108.7376&rep=rep1&type=pdf
    """

    M = -1.0
    epsilon = -1.0

    def __init__(self, initialMedianValue, trainingDataSize=None, epsilon=None):
        self.M = initialMedianValue

        if (not(trainingDataSize is None)):
            self.initialArbitraryValue = trainingDataSize
        else:
            self.initialArbitraryValue = 142 # Size of training data

        if (epsilon is None):
            self.epsilon = max(self.M / 2, self.initialArbitraryValue)
        else:
            self.epsilon = epsilon

        logger.debug(
            """Creating Median Object with the following properties: M = %s, b = %s, epsilon = %s""", str(self.M), str(self.initialArbitraryValue), str(self.epsilon)
        )

    def incrementMedian(self, newValue):
        """
            Calculates the new median incrementally and returns it.
            This is based on the FAME algorithm and Eq.9.

            Parameters
            ----------
                newValue: float
                    The given weight for a quality measure in a new
                    quality profile

            Returns
            -------
                float
                    Returns a new value that represents the incrementally
                    median value.
        """

        newMedian = self.M + self.epsilon * np.sign(newValue - self.M)
        self.M = newMedian
        if abs(newValue - self.M) < self.epsilon:
            self.epsilon = self.epsilon / 2.0

        return self.M

    def getCurrentMedianValue(self):
        return self.M

    def getCurrentEpsilonValue(self):
        return self.epsilon


class Model():
    """
        The purpose of this class is to create a model for a particular
        task using a number of quality profiles that were manually defined by
        participants. Initialising the class would train the model.
        Once trained, propogation methods can be used.
    """

    _medianObjects = dict({})
    modelQualityMeasures = set({})

    def __init__(self, trainingDF=None, taskLabel=None, savedModel=None):
        """
            Constructor for the Training Class.

            Parameters
            ----------
            trainingDF : DataFrame
                A pandas dataframe with the initiating training data.
            taskLabel : String
                (Optional) The label of the task being trained.
        """
        if (not (savedModel is None)):
            logger.info('=== Creating Model for Lenti ===')

            self.taskLabel = savedModel.getTaskLabel()

            self.modelTotalItems = savedModel.getModelTotalItems()
            logger.debug('Number of training quality profiles: ' + str(
                self.modelTotalItems))

            self.totalMetrics = len(savedModel.getMeasureCollection())
            logger.debug('Number of metrics in training set: ' + str(
                self.totalMetrics))

            Model.__loadModel(self, savedModel.getMeasureCollection())
        else:
            logger.info('=== Training and Initial Scoring ===')
            logger.debug('Mapping to a local dataframe')
            self.df = trainingDF

            for i in self.df.columns:
                self.modelQualityMeasures.add(i)

            self.taskLabel = taskLabel

            # Corresponds to m
            self.modelTotalItems = len(self.df.index)
            logger.debug('Number of training quality profiles: ' + str(
                self.modelTotalItems))

            # Corresponds to n
            self.totalMetrics = len(self.df.columns)
            logger.debug('Number of metrics in training set: ' + str(
                self.totalMetrics))

            Model.__trainModel(self)

            Model.__saveModel(self)

    # Private methods to load model from KG
    def __loadModel(self, measureCollection):
        self.modelMeans = {}
        self._medianObjects = {}
        self.modelMedians = {}

        for m in measureCollection:
            # Mean
            self.modelMeans[m.getMeasureLabel()] = m.getMeanValue()
            logger.info('Trained Means: ' + str(self.modelMeans))

            # Median
            self._medianObjects[m.getMeasureLabel()] = _Median(m.getMedianValue(), trainingDataSize=self.modelTotalItems, epsilon=m.getMedianEpsilonValue())
            self.modelMedians[m.getMeasureLabel()] = m.getMedianValue()
            logger.info('Trained Medians: ' + str(self.modelMedians))

            self.modelQualityMeasures.add(m.getMeasureLabel())

        self.trainedTotal = sum([value for key, value in self.modelMeans.items()])
        logger.debug('Trained totals: ' + str(self.trainedTotal))

        self.popularityScores = Model.__calculatePopularityScores(self)
        logger.info('Popularity Scores: ' + str(self.popularityScores))

        # Normalising popularity scores
        self.normalisedPopularityScores = Model.__calculateNorm(self)
        logger.info('Normalised Scores: ' + str(
            self.normalisedPopularityScores))

        logger.info('=== Model Loaded ===')

    # Private methods used to train and update model with new knowledge
    def __trainModel(self):
        # Corresponds to \mu_qm(x) in Eq. 4.4
        self.modelMeans = Model.__calculateMeans(self)
        logger.info('Trained Means: ' + str(self.modelMeans))

        self.modelMedians = Model.__calculateMedians(self)
        logger.info('Trained Medians: ' + str(self.modelMedians))

        # Corresponds to T_\sum(\mu) in Eq. 4.3 - Should add up to 100
        self.trainedTotal = sum([value for key, value in self.modelMeans.items()])
        logger.debug('Trained totals: ' + str(self.trainedTotal))

        # Corresponds to the popularity function in Eq. 4
        self.popularityScores = Model.__calculatePopularityScores(self)
        logger.info('Popularity Scores: ' + str(self.popularityScores))

        # Normalising popularity scores
        self.normalisedPopularityScores = Model.__calculateNorm(self)
        logger.info('Normalised Scores: ' + str(
            self.normalisedPopularityScores))

        # TODO: dispose of dataframe
        logger.info('=== End of Training ===')

    def __calculateMeans(self):
        """
            Calculates the mean value for each column
        """
        _modelMeans = {}
        for i in self.df.columns:
            desc = stats.describe(self.df[i])
            _modelMeans[i] = desc.mean

        return _modelMeans

    def __calculateMedians(self):
        """
            Calculates the median value for each column
        """
        _modelMedians = {}
        for i in self.df.columns:
            medObj = _Median(0)
            self._medianObjects[i] = medObj

            for value in self.df[i]:
                medObj.incrementMedian(value)

            _modelMedians[i] = medObj.getCurrentMedianValue()
            # _modelMedians[i] = np.median(self.df[i])
            # logger.debug("Creating median object for: "+ str(i))
            # medObj = _Median(_modelMedians[i])
            # self._medianObjects[i] = medObj

        return _modelMedians

    def __calculatePopularityScores(self):
        """
            Calculates the popularity score for each metric
        """
        p_0 = 1.0/self.totalMetrics
        sigma = math.sqrt((p_0*(1.0-p_0))/self.modelTotalItems)

        popularityScores = dict({})

        for key, value in self.modelMeans.items():
            p_hat = value / self.trainedTotal
            z_score = (p_hat - p_0) / sigma
            popularityScores[key] = z_score

        return popularityScores

    def __calculateNorm(self):
        """
            Calculates the normalisation score of each metric
            based on their popularity score.
        """
        normalisedPopularityScore = dict({})
        for key, value in self.popularityScores.items():
            normalisedPopularityScore[key] = norm.sf(value)

        return normalisedPopularityScore

    def __boundaryWeights(self, measureWeights):
        """
            Normalise the weights of the recommended quality measures
            between 0 and 1 both inclusive.
        """
        if (sum([value for key, value in measureWeights.items()]) == 1.0):
            return measureWeights

        total = sum([value for key, value in measureWeights.items()])
        for key, value in measureWeights.items():
            measureWeights[key] = (value * 1.0) / total
            logger.debug(
                "Normalised weight for %s from: %s to: %s",
                str(key), str(value), str(measureWeights[key])
            )
        return measureWeights

    def __incrementalMean(self, qualityProfile):
        """
            Increments the model's mean table based on a new quality profile.

            Parameter
            ---------
                qualityProfile : dict
                    The quality profile, where the key is the quality measure
                    and the value is the given weight by the consumer
        """
        __newMeanValues = {}
        self.modelTotalItems += 1

        for key, value in qualityProfile.items():
            oldMean = self.modelMeans[key]
            __newMeanValues[key] = oldMean + ((value - oldMean) / self.modelTotalItems)

        return __newMeanValues

    def __incrementalMedian(self, qualityProfile):
        """
            Increments the model's median table based on a new quality profile.

            Parameter
            ---------
                qualityProfile : dict
                    The quality profile, where the key is the quality measure
                    and the value is the given weight by the consumer
        """
        __newMedianValues = {}
        for key, value in qualityProfile.items():
            __newMedianValues[key] = self._medianObjects[key].incrementMedian(value)

        return __newMedianValues

    def __saveModel(self):
        qualityMeasures = []
        for k, v in self.modelMeans.items():
            measure = QualityMeasure(k,
                                     "http://example.org/" + str(k), v,
                                     self._medianObjects[k].getCurrentMedianValue(),
                                     self._medianObjects[k].getCurrentEpsilonValue())
            qualityMeasures.append(measure)

        saveModel = SavedModel(self.taskLabel, self.modelTotalItems, qualityMeasures)

        ds = Datastore() # Wire global datastore or singleton
        ds.createTrainedModel(saveModel, "")

    def __updateModel(self):
        qualityMeasures = []
        for k, v in self.modelMeans.items():
            measure = QualityMeasure(k,
                                     "http://example.org/" + str(k), v,
                                     self._medianObjects[k].getCurrentMedianValue(),
                                     self._medianObjects[k].getCurrentEpsilonValue())
            qualityMeasures.append(measure)

        saveModel = SavedModel(self.taskLabel, self.modelTotalItems, qualityMeasures)

        ds = Datastore() # Wire global datastore or singleton
        ds.updateKnowledgeGraph(saveModel, "")

    # Public Methods
    def updateModel(self, qualityProfile):
        """
            Updates the model with a new quality profile for the task.

            Parameters
            ----------
                qualityProfile : dict
                    The quality profile, where the key is the quality measure
                    and the value is the given weight by the consumer.

            Returns
            -------
                bool
                    Returns True if model is updated successfully.
        """

        # Populate quality profile with 0s for missing Measures
        for i in self.modelQualityMeasures:
            if (not(i in qualityProfile)):
                qualityProfile[i] = 0

        # Increment Model Mean Values - Eq. 7
        self.modelMeans = Model.__incrementalMean(self, qualityProfile)
        logger.debug('Updated Means: ' + str(self.modelMeans))

        # Corresponds to T_\sum(\mu) in Eq. 4.3 - This should add up to 100
        self.trainedTotal = sum([value for key, value in self.modelMeans.items()])
        # TODO: raise error if trainedTotal does not add up to 100
        logger.debug('Means totals: ' + str(self.trainedTotal))

        # Increment Model Median Values - Eq. 7
        self.modelMedians = Model.__incrementalMedian(self, qualityProfile)
        logger.debug('Updated Medians: ' + str(self.modelMedians))

        # Corresponds to the popularity function in Eq. 4
        self.popularityScores = Model.__calculatePopularityScores(self)
        logger.debug('Updated Popularity Scores: ' + str(self.popularityScores))

        # Normalising popularity scores
        self.normalisedPopularityScores = Model.__calculateNorm(self)
        logger.debug('Updated Normalised Scores: ' + str(self.normalisedPopularityScores))

        self.__updateModel()
        return True

    def recommendQualityMeasures(self, threshold=None):
        """
            Provides a set of quality measures that are important based on
            a given threshold (statistical significance) value. This value
            is used in order to accept or reject the hypothesis defined in
            the paper.

            By default the threshold is set to 0.01.

            Parameters
            ----------
                threshold : float
                    (Optional) The statistical significance value
                    to accept or reject the hypothesis test. The value
                    should be between 0 and 1 (both inclusive).
                    The default value is 0.01.

            Returns
            -------
                dict
                    A set of recommended quality metrics and weights that
                    pass the hypothesis test, i.e. that are important for
                    a task based on the given threshold. This corresponds
                    to M_T in Eq. 5 with the weight defined according to
                    the function imp(qm_x) in Eq. 6.
        """
        if (threshold is None):
            threshold = 0.01
        if (not(type(threshold) is float)):
            logger.error("""The threshold parameter should be a float value between 0 and 1 (both inclusive). Given parameter type is: """ + type(threshold).__name__)
            raise ValueError("""The threshold parameter should be a float value. Passed type is: """ + type(threshold).__name__)
            threshold = 0.01
        if ((threshold < 0) or (threshold > 1)):
            logger.error("""The threshold parameter should be a float value between 0 and 1 (both inclusive). Given parameter value is: """ + threshold)
            raise ValueError("""The threshold parameter should be a float value between 0 and 1 (both inclusive). Given parameter value is: """ + type(threshold).__name__)
            threshold = 0.01

        currentModel = dict({})
        for key, value in self.normalisedPopularityScores.items():
            if (value < threshold):
                # there is strong evidence to accept the null hypothesis
                # hence the quality measure is important for this task.
                logger.debug("Measure: " + str(key) + " is important")
                currentModel[key] = self.modelMedians[key]

        logger.info("Chosen measures with median weights: " + str(currentModel))
        recommendedMeasures = self.__boundaryWeights(currentModel)
        return recommendedMeasures
