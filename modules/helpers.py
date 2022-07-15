
from statistics import variance
import string
import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import bct
from matplotlib import pyplot as plt
import seaborn as sns
import bct

from typing import List, Dict

from nilearn.image import load_img, index_img
from scipy import io, stats
from matplotlib import pyplot as plt


def readFromCSV(filePath: str):
    fileData = pd.read_csv(filePath)
    return fileData

def loadMatFile(filePath, fileName):
    componentDict = io.loadmat(filePath+fileName)
    return componentDict


# calculation Pearson CrossCorrelation
def calculatePearsonCrossCorrelations(listA: list, listB: list):
        xmean = np.mean(listA)
        xval = listA - xmean
        xsqu = np.sqrt(np.sum(np.square(xval)))

        ymean = np.mean(listB)
        yval = listB - ymean
        ysqu = np.sqrt(np.sum(np.square(yval)))

        num = 0
        for k in range(len(yval)):
            num += (xval[k]*yval[k])

        den = (xsqu * ysqu)
        ans = num/den
        return ans


# Functions for 4D Images:
# 1. loading 4D image from .nii file
def load4DImageData(imagePath):
    return load_img(imagePath)

# 2 Getting Image at a particular timecourse
def getTimeCourseImage(spatialImage,index):
    return index_img(spatialImage, index)

# 3 Getting 3D image data
def getImageData(componentImage):
    return componentImage.getf_data()

# Calculating Voxel Count for an 3D-Image
def countVoxels(imageData):
    image_threshold = 3*np.std(imageData)
    voxelCount = np.count_nonzero(imageData > image_threshold)
    return voxelCount

# Computing Graph Metrics

### Global Graph Metrics
def computeGlobalEfficiency(fncMatrix):
    globalEfficiency = nx.global_efficiency(fncMatrix)
    return globalEfficiency

def computeCharacteristicPathLength(fncMatrix):
    characteristicPathLength = nx.average_shortest_path_length(fncMatrix, weight='weight')
    return characteristicPathLength

def computeClusteringCoefficient(fncMatrix):
    clusterCofficient = nx.clustering(fncMatrix, weight='weight')
    clusterCofficient_network = 0
    for i in clusterCofficient:
        clusterCofficient_network += clusterCofficient[i]
    clusterCofficient_network /= 53
    return clusterCofficient_network

### Component Metrics
def computeDegree(fncMatrix):
    all_nodes_degree = nx.degree(fncMatrix, weight= 'weight')
    return all_nodes_degree

def computeClosenessCentrality(fncMatrix):
    all_nodes_cc = nx.closeness_centrality(fncMatrix, distance='weight', wf_improved=False)
    return all_nodes_cc

def computeParticipationCoefficient(fncMatrix):
    fncMatrix_numpy = nx.to_numpy_array(fncMatrix)
    modularity = nx.algorithms.community.greedy_modularity_communities(fncMatrix)
    sam=[]
    for i in range(len(modularity)):
        for j in list(modularity[i]):
            sam.append(j)
    sam = np.array(sam)
    all_nodes_pc = bct.centrality.participation_coef(fncMatrix_numpy, ci=sam, degree='undirected')
    return all_nodes_pc


## File operations

## Saving a ndarray
def saveNdarray(arr: np.ndarray, name):
    a_file = open(name, "w")
    strData = ""
    row,column = arr.shape
    for i in range(row):
        for j in range(column):
            strData+=str(arr[i][j])+" "
        strData+="\n"
    a_file.write(strData)
    a_file.close()

# FNC Operations

## Prepare FNC from Timecourse data:
def prepareFNCArray(componentData, indexList):
    selected_component = np.zeros((53,53), dtype=np.float64).reshape(53,53)
    for i in range(53):
        for j in range(i+1, 53):
            # if componentData[self.indexList[i]-1][self.indexList[j]-1] >=0 : 
            selected_component[i][j]=componentData[indexList[i]-1][indexList[j]-1]
    selected_component += selected_component.T

    # # Finding correlation matrix
    corrs = pd.DataFrame(selected_component)
    correlation_matrix = corrs.corr()
    correlation_numpy_array = correlation_matrix.to_numpy(dtype=np.float64)
    return correlation_numpy_array

## Threshold the FNC Array
def thresholdFNCArray(fncArray):
        # Finding Thresholded correlation array
        row,column = fncArray.shape
        for i in range(row):
            for j in range(column):
                if i!=j and fncArray[i][j]<0:
                    fncArray[i][j]=0
        return fncArray

# Convert Numpy Array to Numpy Matrix
def numpyMatrix(fncNumpyArray):
        input_matrix = nx.from_numpy_array(fncNumpyArray)
        return input_matrix

# Prepare HeapMap for Matrix
def prepareHeatmap(xlabels, ylabels, component, name, vmin=-1.00, vmax=1.00):
    fig, ax = plt.subplots(figsize=(8,8))
    sd = sns.heatmap(
            component, cmap='coolwarm', square=True,ax=ax, linewidth='0.8', center=0, annot_kws={"size": 3},
            xticklabels=xlabels, yticklabels=xlabels, annot=False, vmax=vmax, vmin=vmin,
            cbar_kws={"fraction": 0.046, "pad": 0.02}
        )

    sd.set_yticklabels(labels= xlabels,fontsize=3, weight='bold')
    sd.set_xticklabels(labels= ylabels,fontsize=3, weight='bold')
    
    ax.set_title(name+'.png')
    ax.tick_params(axis='both', which='major', labelsize=10,width=3, color=(0,0,0))
    ax.tick_params(axis='both', which='minor', labelsize=10,width=3, color=(0,0,0))
    plt.savefig(name+'.png', dpi=1200)


### Calculate P-Values
def calculatePvalue(corrs, size, out=False):

    # Formula for t-value: r*sqrt((n-2)/(1-r^2))
    num = (size-2)
    den = (1 - pow(corrs,2))
    val = num/den
    t_val = corrs * np.math.sqrt(val)

    # function for p-value
    p_val = 2*(1 - stats.t.cdf(abs(t_val), num))
    if out:
        print("t-val:", t_val, " corrs:", corrs, " p-val:", p_val, " size:", size)
    return p_val

def getGlobalGraphMetricsForAvgFNCs(graphObj):

    x = list(graphObj.groupsToCompute)
    y = list()
    metricNames = graphObj.graphMetricGlobalMeasues

    for subjectName in x:
        metricList=list()
        avgFNCMatrix = numpyMatrix(graphObj.avgFNCs[subjectName])
        metricList.append(computeGlobalEfficiency(avgFNCMatrix))
        metricList.append(computeCharacteristicPathLength(avgFNCMatrix))
        metricList.append(computeClusteringCoefficient(avgFNCMatrix))

        metricList = [ round(i,3) for i in metricList ]
        # print(metricList)
        y.append(metricList)

    barPos = list()
    barWidth = 0.25
    for i in range(len(y)):
        if i==0:
            barPos.append(np.arange(len(y[i])))
        else:
            barList = [ dist + barWidth for dist in barPos[i-1] ]
            barPos.append(barList)
    print(x,y)

    fig = plt.figure(figsize = (15, 10))
    color = ['r','g','b']
    for i in range(len(y)):
        plt.bar(barPos[i], y[i], color=color[i], width=barWidth, edgecolor='grey', label=x[i])

    # plt.figure(figsize=(6,4))
    plt.xlabel('Graph Metric Names', fontweight='bold', fontsize=15)
    plt.ylabel('Graph Metirc Values', fontweight='bold', fontsize=15)
    plt.xticks([r+barWidth for r in range(len(y[0]))], metricNames, fontsize=15, weight='bold')
    plt.yticks([i/10 for i in range(0, 11)], fontsize=15, weight='bold')
    plt.legend(prop={"size": 20})
    plt.savefig('Global Metrics', dpi=1200)


def getComponentGraphMetricsForAvgFNCs(graphObj):

    subjectList = graphObj.groupsToCompute
    # for component measures
    x = list()
    for ind in range(len(graphObj.indexList)):
        x.append(graphObj.domainList[ind]+'('+str(graphObj.indexList[ind])+')')
        if(ind != 0 and graphObj.domainList[ind] == graphObj.domainList[ind-1]):
            x[ind]='('+str(graphObj.indexList[ind])+')'
    
    for subjectName in subjectList:
        for graphMetric in graphObj.graphMetricComponentMeasures:
            y = list()
            avgFNCMatrix = numpyMatrix(graphObj.avgFNCs[subjectName])
            dictObj = None

            if graphMetric == "Degree":
                dictObj = computeDegree(avgFNCMatrix)

            elif graphMetric == "Closeness centrality":
                dictObj = computeClosenessCentrality(avgFNCMatrix)

            elif graphMetric == "Participation coefficient":
                dictObj = computeParticipationCoefficient(avgFNCMatrix)

            else:
                print("Not Found: ", graphMetric)
                assert(False)
            
            
            for i in range(len(x)):
                y.append(dictObj[i])

            fig = plt.figure(figsize = (90, 10))
            plt.margins(0)
            plt.bar(x, y, color ='blue',
                    width = 0.3)
            plt.xlabel('Components in brain Network', fontweight='bold', fontsize=40)
            plt.ylabel('Graph Metirc Values', fontweight='bold', fontsize=40)
            plt.xticks(fontsize=30, weight='bold')
            plt.yticks(fontsize=30, weight='bold')
            plt.title(subjectName+'-'+graphMetric, fontsize=40, weight='bold')
            plt.savefig(subjectName+'-'+graphMetric)

## Compute difference between the Avg FNC matrix significance test.
def computeDiffBetweenAvgFNCs(subjectToCompute: 'list', avgFNCs: 'dict[int, np.ndarray]') -> list:
    diffFNCArrays = list()
    subjectKeys = list()
    # print(subjectToCompute)
    for sub1 in range(len(subjectToCompute)):

        for sub2 in range(sub1+1, len(subjectToCompute)):

            subject1 : str = subjectToCompute[sub1]
            subject2 : str = subjectToCompute[sub2]

            mat1 : np.ndarray = avgFNCs[subject1]
            mat2 : np.ndarray = avgFNCs[subject2]

            diffMat = (mat1 - mat2)
            diffArray = diffMat.flatten()

            diffKey = subject1 + '-' + subject2

            avgFNCMatrix = numpyMatrix(diffMat)

            print(diffKey+" : ")
            print("Global Efficiency: ", computeGlobalEfficiency(avgFNCMatrix))
            print("CharacterisitcPathLength: ", computeCharacteristicPathLength(avgFNCMatrix))
            print("Clustering Cofficient", computeClusteringCoefficient(avgFNCMatrix))

            subjectKeys.append(diffKey)
            diffFNCArrays.append(diffArray)

    print(subjectKeys)
    pvalues = []
    for i in range(len(diffFNCArrays)):
        for j in range(i+1, len(diffFNCArrays)):

            crossCorrelation = calculatePearsonCrossCorrelations(diffFNCArrays[i], diffFNCArrays[j])
            print(crossCorrelation)
        
            pvalues.append(calculatePvalue(crossCorrelation, len(diffFNCArrays[i])))


def two_Sample_t_test(arr1: np.ndarray, arr2: np.ndarray):
    # Follow Link: https://www.statology.org/two-sample-t-test-python/
    
    equal_variance = False
    variance1, variance2 = np.var(arr1), np.var(arr2)

    populationVariance = variance1 / variance2
    print("Varience Ratio: ", populationVariance)

    if populationVariance < 4:
        equal_variance = True
    
    print("equal_variance: ", equal_variance)

    results = stats.ttest_ind(arr1, arr2, equal_var=equal_variance)
    print("results: ", results)

    pass