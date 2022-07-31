from modules.subject import GroupsInfo
from modules.helpers import *

import numpy as np
import networkx as nx
import seaborn as sns
import bct
from matplotlib import pyplot as plt
import seaborn as sns
import bct
import json

from matplotlib import pyplot as plt
from copy import deepcopy

import traceback

class GraphMetrics:

    " \
        datasetPath : 4D images path \
        componentsPath: indexes path \
        subjectList: List of subjects to calculate ex: AD, CN, ....  \
    "
    graphMetrics = None
    def __init__(self):

        ## Data coming from subject.ipynb
        self.subjectsData = GroupsInfo.getInstance()
        self.groupsToCompute = self.subjectsData.getGroupDictionary().keys()
        self.indexList = self.subjectsData.getICNIndexes()
        self.domainList = self.subjectsData.getICNDomainNames()
        self.graphMetricGlobalMeasues = self.subjectsData.getGlobalGraphMetrics()
        self.graphMetricComponentMeasures = self.subjectsData.getComponentGraphMetrics()
        self.FNCmatFile = self.subjectsData.getFNCFileName()
        self.component_key = self.subjectsData.getComponentKey()
        self.perPT = self.subjectsData.perPT

        self.graphMetricMap = dict()
        self.avgFNCs : 'dict[str, np.ndarray]' = dict()
        self.avgDiffFNCs : 'dict[str, np.ndarray]' = dict()
        self.dataSetLength = dict()
        self.overallFNC = np.zeros((53,53), dtype=np.float64).reshape(53,53)
        self.prepareGraphMetricMap()

    def prepareGraphMetricMap(self):
        for subjectName in self.groupsToCompute:
            self.graphMetricMap[subjectName]=dict()
            self.graphMetricMap[subjectName]["paths"] = self.subjectsData.getGroupDatasetPaths(subjectName)
            self.dataSetLength[subjectName] = len(self.graphMetricMap[subjectName]["paths"])
            for graphMetric in self.graphMetricGlobalMeasues:
                self.graphMetricMap[subjectName][graphMetric]=list()

            for graphMetric in self.graphMetricComponentMeasures:
                self.graphMetricMap[subjectName][graphMetric] = dict()
                for ind in self.indexList:
                    self.graphMetricMap[subjectName][graphMetric][ind]=list()
            
            # Avg FNC Matrix of all subjects for each group
            self.avgFNCs[subjectName] = deepcopy(np.zeros((53,53), dtype=np.float64).reshape(53,53))

    def calculateGraphMetrics(self):
        logging.info("<<< calculateGraphMetrics ------")
        failSubjects = dict()
        failSubjects['CN'] = 0
        failSubjects['MCI'] = 0
        failSubjects['AD'] = 0

        for subjectName in self.groupsToCompute:
            logging.info('***********subjectName: '+ subjectName+ ' ************')
            for path in self.graphMetricMap[subjectName]["paths"]:
                # logging.info('\n'+path)
                # Preparing FNC Matrix:
                ## 1. Get timecourse data from .mat file..
                timecourseData = loadMatFile(path)
                
                ## 2. Create FNCArray from the timcourseData
                fncMatrixArray = prepareFNCArray(timecourseData, self.indexList)

                ## 3. threshold the above numpy array for negative values.
                thresholdedFNCArray = thresholdFNCArray(fncMatrixArray, perPT=self.perPT)

                ## 4. Convert the fncArray to fncMatrix for graph algo..
                fncMatrix = numpyMatrix(thresholdedFNCArray)

                # for global measures:
                updateFailureOnce = True
                for graphMetric in self.graphMetricGlobalMeasues:
                    try:
                        if graphMetric == "Global efficiency":
                            globalEfficieny = computeGlobalEfficiency(fncMatrix)
                            self.graphMetricMap[subjectName][graphMetric].append(globalEfficieny)

                        elif graphMetric == "Characteristic path length":
                            characteristicPathLength = computeCharacteristicPathLength(fncMatrix)
                            self.graphMetricMap[subjectName][graphMetric].append(characteristicPathLength)

                        elif graphMetric == "Clustering coefficient":
                            clusterCofficient_network = computeClusteringCoefficient(fncMatrix)
                            self.graphMetricMap[subjectName][graphMetric].append(clusterCofficient_network)

                        else:
                            print("Not Found: ", graphMetric)
                            assert(False)
                    except:
                        str1 = "\nGraph Metric: " + graphMetric
                        logging.info(str1)
                        if updateFailureOnce:
                            failSubjects[subjectName] += 1
                            updateFailureOnce=False

                # for component measures
                updateFailureOnce = True
                for graphMetric in self.graphMetricComponentMeasures:
                    try:
                        if graphMetric == "Degree":
                            all_nodes_degree = computeDegree(fncMatrix)
                            for ind in range(len(self.indexList)):
                                self.graphMetricMap[subjectName][graphMetric][self.indexList[ind]].append(all_nodes_degree[ind])

                        elif graphMetric == "Closeness centrality":
                            all_nodes_cc = computeClosenessCentrality(fncMatrix)
                            for ind in range(len(self.indexList)):
                                self.graphMetricMap[subjectName][graphMetric][self.indexList[ind]].append(all_nodes_cc[ind])

                        elif graphMetric == "Participation coefficient":
                            all_nodes_pc = computeParticipationCoefficient(fncMatrix)
                            for ind in range(len(self.indexList)):
                                self.graphMetricMap[subjectName][graphMetric][self.indexList[ind]].append(all_nodes_pc[ind])

                        else:
                            print("Not Found: ", graphMetric)
                            assert(False)
                    except:
                        str1 = "\nGraph Metric: " + graphMetric
                        logging.info(str1)
                        if updateFailureOnce:
                            failSubjects[subjectName] += 1
                            updateFailureOnce=False
        logging.info('\n*** Failed Subjects ****\n'+json.dumps(failSubjects))

    def calAvgAllGroupFNC(self, dest=""):
        totalLength = 0
        domainNames = self.modDomainNames()
        for subjectName in self.groupsToCompute:
            dataSetPaths = self.graphMetricMap[subjectName]["paths"]
            totalLength += len(dataSetPaths)

            # for path in dataSetPaths:
            #     fncArray = thresholdFNCArray(prepareFNCArray(loadMatFile(path), self.indexList), self.perPT)
            #     self.avgFNCs[subjectName] = np.add( self.avgFNCs[subjectName] , fncArray)

            timeSeriesData = np.zeros((53,53), dtype=np.float64).reshape(53,53)
            
            n = len(self.indexList)
            for path in dataSetPaths:
                componentData = loadMatFile(path)
                for i in range(n):
                    for j in range(n):
                        timeSeriesData[i][j] += componentData[self.indexList[i]-1][self.indexList[j]-1]
            
            timeSeriesData /= len(dataSetPaths)

            for i in range(n):
                for j in range(n):
                    if i==j:
                        self.avgFNCs[subjectName][i][j] = 0
                    else:
                        correlationTimeSeries = calculatePearsonCrossCorrelations(timeSeriesData[i], timeSeriesData[j])
                        self.avgFNCs[subjectName][i][j] = correlationTimeSeries if correlationTimeSeries >= self.perPT else 0

            # self.overallFNC = np.add(self.overallFNC, self.avgFNCs[subjectName])
            # self.avgFNCs[subjectName] = self.avgFNCs[subjectName] / len(dataSetPaths)
            
            prepareHeatmap(
                domainNames,
                domainNames,
                self.avgFNCs[subjectName],
                subjectName+' FNC',
                dest,
                vmin=0,
                vmax=1
            )
        # self.overallFNC = self.overallFNC / totalLength
    def modDomainNames(self):
        filteredTicks = deepcopy(self.domainList)
        for i in range(1, len(self.domainList)):
            if self.domainList[i] == self.domainList[i-1]:
                filteredTicks[i]=''
        return filteredTicks

    def calDiffAvgGroupFNC(self, path=""):
        domainNames = self.modDomainNames()
        subjects = list(self.groupsToCompute)
        for sub1 in range(len(subjects)):
            for sub2 in range(sub1+1, len(subjects)):
                subject1 = subjects[sub1]
                subject2 = subjects[sub2]

                mat1 = deepcopy(self.avgFNCs[subject1])
                mat2 = deepcopy(self.avgFNCs[subject2])

                diffMat = np.zeros((53,53), dtype=np.float64).reshape(53,53)
                row , col = diffMat.shape 
                for i in range(row):
                    for j in range(col):
                        diffVal = mat1[i][j] - mat2[i][j]
                        diffMat[i][j] = diffVal
                
                self.avgDiffFNCs[subject1+' Vs '+subject2] = diffMat

                prepareHeatmap(
                    domainNames,
                    domainNames,
                    diffMat,
                    subject1+' Vs '+subject2+' Difference FNC',
                    path,
                    vmin=-1,
                    vmax=1
                )

    def subjectList(self):
        x = list()
        for i in range(len(self.indexList)):
            if(i!=0 and self.domainList[i] == self.domainList[i-1]):
                x.append('('+str(self.indexList[i])+')')
            else:
                x.append(self.domainList[i]+'('+str(self.indexList[i])+')')
        return x

    def calculateCrossCorrelationOfAvgFNC(self):
        self.prepareAvgFNC()
        self.calAvgAllGroupFNC()
        
        fig = plt.figure(figsize = (50, 15))
        # pt.ylim(0.900,1.000)
        plt.xlim(100, 400)
        # x = self.subjectList()
        color = dict({
                    'CN': 'red', 
                    'MCI': 'blue',
                    'AD': 'green'
                })

        self.overallFNC = self.overallFNC.flatten()
        for subjectName in self.groupsToCompute:
            y = list()
            x = list()
            dataSetPaths = self.graphMetricMap[subjectName]["paths"]
            x = [i+1 for i in range(len(dataSetPaths))]
            print("subjectName: ", subjectName, "  Total:", len(dataSetPaths))
            for path in dataSetPaths:
                fncArray = prepareFNCArray(loadMatFile(path))
                listA = fncArray.flatten('C')
                listB = deepcopy(self.overallFNC)
                crossCorr = self.calculatePearsonCrossCorrelations(listA, listB)
                y.append(crossCorr)

            # creating the bar plot
            # print(min(y), max(y))
            plt.plot(x, y, color = color[subjectName], label=subjectName)
        plt.margins(0)
        plt.xticks(fontsize=20, weight='bold')
        plt.yticks(fontsize=20, weight='bold')
        plt.xlabel("Group Names", fontweight='bold', fontsize=15)
        plt.ylabel("Avg Correlation of FNC", fontweight='bold', fontsize=15)
        plt.legend(prop={"size":30})
        plt.title("Correlation of Avg Group FNC to Overall FNC")
        plt.savefig("avg", dpi=600)

    def prepareHeatmap(self, component, name, vmin=-1.00, vmax=1.00):

        filteredTicks = deepcopy(self.domainList)
        for i in range(1, len(self.domainList)):
            if self.domainList[i] == self.domainList[i-1]:
                filteredTicks[i]=''

        fig, ax = plt.subplots(figsize=(8,8))
        sd = sns.heatmap(
                component, cmap='coolwarm', square=True,ax=ax, linewidth='0.8', center=0, annot_kws={"size": 3},
                xticklabels=filteredTicks, yticklabels=filteredTicks, annot=False, vmax=vmax, vmin=vmin,
                cbar_kws={"fraction": 0.046, "pad": 0.02}
            )

        sd.set_yticklabels(labels= filteredTicks,fontsize=3, weight='bold')
        sd.set_xticklabels(labels= filteredTicks,fontsize=3, weight='bold')

        ax.set_title(name+'.png')
        ax.tick_params(axis='both', which='major', labelsize=10,width=3, color=(0,0,0))
        ax.tick_params(axis='both', which='minor', labelsize=10,width=3, color=(0,0,0))
        plt.savefig(name+'.png', dpi=1200)

    ####################
    # Getter Functions #
    ####################

    @classmethod
    def getInstance(cls):
        if cls.graphMetrics == None:
            cls.graphMetrics = cls()
        return cls.graphMetrics

    def getGlobalGraphMetricValues(self, subjectName, graphMetricName):
        if graphMetricName in self.graphMetricGlobalMeasues:
            return deepcopy(self.graphMetricMap[subjectName][graphMetricName])
        return list()

    def getComponentGraphMetricIndValues(self, subjectName, graphMetricName, ind):
        if graphMetricName in self.graphMetricComponentMeasures:
            return deepcopy(self.graphMetricMap[subjectName][graphMetricName][ind])
        return list()
