
import logging
from math import sqrt
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
from matplotlib.colors import ListedColormap

import logging


def readFromCSV(filePath: str):
    fileData = pd.read_csv(filePath)
    return fileData

def loadMatFile(filePath, fileName="adni_aa__postprocess_results.mat", componentKey="fnc_corrs_all"):
    componentDict = io.loadmat(filePath+fileName)
    return componentDict[componentKey]

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
        return round(ans,2)


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
    characteristicPathLength = nx.average_shortest_path_length(fncMatrix)
    return characteristicPathLength

def computeClusteringCoefficient(fncMatrix):
    clusterCofficient = nx.average_clustering(fncMatrix)
    return clusterCofficient

### Component Metrics
def computeDegree(fncMatrix):
    all_nodes_degree = nx.degree(fncMatrix)
    return all_nodes_degree

def computeClosenessCentrality(fncMatrix):
    all_nodes_cc = nx.closeness_centrality(fncMatrix)
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
def saveNdarray(arr: np.ndarray, name: str = "ndArray"):
    logging.info(">>> saveNdarray -------")
    strData = ""
    row,column = arr.shape
    for i in range(row):
        for j in range(column):
            strData+=str(arr[i][j])+" "
        strData+="\n"
    
    logging.info("name: %s, \narr: %s", name, strData )
    logging.info(">>> saveNdarray -------")

# FNC Operations

## Prepare FNC from Timecourse data:
def prepareFNCArray(componentData, indexList):
    selected_component:np.ndarray = np.zeros((53,53), dtype=np.float64).reshape(53,53)
    n = len(indexList)
    for i in range(n):
        for j in range(i+1, n):
            # if componentData[self.indexList[i]-1][self.indexList[j]-1] >=0 : 
            selected_component[i][j]=componentData[indexList[i]-1][indexList[j]-1]
    selected_component += selected_component.T

    # # # Finding correlation matrix
    corrs = pd.DataFrame(selected_component)
    correlation_matrix = corrs.corr()
    correlation_numpy_array = correlation_matrix.to_numpy(dtype=np.float64)
    return correlation_numpy_array

## Threshold the FNC Array
def thresholdFNCArray(fncArray, perPT=0):
        # Finding proportional Thresholded correlation array
        row,column = fncArray.shape
        for i in range(row):
            for j in range(column):
                if i == j:
                    fncArray[i][j] = 0
                else:
                    fncArray[i][j] = fncArray[i][j] if fncArray[i][j] >= perPT else 0
        return fncArray

# Convert Numpy Array to Numpy Matrix
def numpyMatrix(fncNumpyArray):
        input_matrix = nx.from_numpy_array(fncNumpyArray)
        return input_matrix

def prepareHeatMapObj(
    data, *,
    vmin=None, vmax=None, cmap=None, center=None, robust=False,
    annot=None, fmt=".2g", annot_kws=None,
    linewidths=0, linecolor="white",
    cbar=True, cbar_kws=None, cbar_ax=None,
    square=False, xticklabels="auto", yticklabels="auto",
    mask=None, ax=None,
    **kwargs
):
    pass

# Prepare HeapMap for Matrix
def prepareHeatmap(
    xlabels, 
    ylabels, 
    component, 
    name,
    path: str="",
    vmin=-1.00, 
    vmax=1.00,
    figsize=(8,8)
):
    fig, ax = plt.subplots(figsize=figsize)
    sd = sns.heatmap(
            component, cmap='coolwarm', square=True,ax=ax, linewidth='0.8', center=0, annot_kws={"size": 3},
            xticklabels=xlabels, yticklabels=ylabels, annot=False, vmax=vmax, vmin=vmin,
            cbar_kws={"fraction": 0.046, "pad": 0.02}
        )

    sd.set_yticklabels(labels= ylabels,fontsize=3, weight='bold')
    sd.set_xticklabels(labels= xlabels,fontsize=3, weight='bold')
    
    ax.set_title(name)
    ax.tick_params(axis='both', which='major', labelsize=10,width=3, color=(0,0,0))
    ax.tick_params(axis='both', which='minor', labelsize=10,width=3, color=(0,0,0))
    plt.savefig(
        path+name+'.png',
        pad_inches = 1,
        transparent = True,
        facecolor ="w",
        edgecolor ='g',
        orientation ='landscape',
        format='png',
        dpi=1200
    )
    plt.close()


### Calculate P-Values
def calculatePvalue(corrs, size, out=False):
    logging.info("<<<< CalculatePvalue -------")
    # Formula for t-value: r*sqrt((n-2)/(1-r^2))
    num = (size-2)
    den = (1 - pow(corrs,2))
    val = num/den
    t_val = corrs * np.math.sqrt(val)

    # function for p-value
    p_val = 2*(1 - stats.t.cdf(abs(t_val), num))
    if out:
        logging.info("t-val: %s, corrs: %s, p-val: %s, size: %s", t_val, corrs, p_val, size)
    logging.info(">>> CalculatePvalue -------")
    return p_val

def getGlobalGraphMetricsForAvgFNCs(graphObj, path="", name="None"):

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
    plt.ylabel('Graph Metric Values', fontweight='bold', fontsize=15)
    plt.xticks([r+barWidth for r in range(len(y[0]))], metricNames, fontsize=15, weight='bold')
    plt.yticks([i/10 for i in range(0, 21, 5)], fontsize=15, weight='bold')
    plt.legend(prop={"size": 20})
    plt.title(name)
    plt.savefig(
        path+name+'.png',
        pad_inches = 1,
        transparent = True,
        facecolor ="w",
        edgecolor ='g',
        orientation ='landscape',
        format='png',
        dpi=1200
    )
    plt.close()


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


def two_Sample_t_test(arr1: np.ndarray, arr2: np.ndarray,mean_1: np.float ,mean_2: np.float ) -> tuple:
    # Follow Link: https://www.investopedia.com/terms/t/t-test.asp#:~:text=The%20t%2Dvalue%2C%20or%20t,of%20the%20two%20sample%20sets.
    # Under : Unequal Variance T-Test
    """
        t-value = (mean1 - mean2)/(sqrt( ( (var1/n1) + (var2/n2) ) ) )
        p-value = cipy.stats.t.sf(abs(t_score), df=degree_of_freedom)
        df = x1 / x2
        x1 = (y1+y2)^2
        x2 = ( ((y1)^2)/n1-1 + ((y2)^2)/n2-1 )
        y1 = (var1^2)/n1
        y2 = (var2^2)/n2
    """
    logging.info("<<< two_Sample_t_test ------")

    n_1, n_2  = len(arr1), len(arr2)
    logging.warning("size1: %s, size2: %s", n_1, n_2)

    # var_1, var_2 = round(np.var(arr1), 3), round(np.var(arr2), 3)
    var_1, var_2 = np.var(arr1), np.var(arr2)
    logging.info("var1: %s, var2: %s", var_1, var_2)

    s1, s2 = var_1/n_1, var_2/n_2
    # mean_1, mean_2 = round(np.mean(arr1), 3), round(np.mean(arr2), 3)
    logging.info("mean1: %s, mean2: %s", mean_1, mean_2)

    t_value_numerator = mean_1 - mean_2
    t_value_denominator = sqrt((s1+s2))

    t_value = round(t_value_numerator/t_value_denominator, 3)
    logging.info("t-value: %s", t_value)

    y1, y2 = pow(var_1, 2)/n_1, pow(var_2, 2)/n_2
    df_value_numerator = pow(y1+y2, 2)

    z1, z2 = pow(y1, 2)/(n_1-1), pow(y2, 2)/(n_2-1)
    df_value_denominator = z1 + z2

    df_value = df_value_numerator / df_value_denominator
    logging.info("df: %s", df_value)

    p_value = 2*stats.t.sf(abs(t_value), df_value)
    logging.info("p-val: %s", p_value)

    logging.info(">>> two_Sample_t_test ------")
    return t_value, p_value


"""DataFrame"""
def prepareDataFrame(indexList, domainList):
    dummyDataFrame = pd.DataFrame()
    for i in range(len(indexList)):
        if(i!=0 and domainList[i] == domainList[i-1]):
            dummyDataFrame['('+str(indexList[i])+')'] = list()
        else:
            dummyDataFrame[domainList[i]+'('+str(indexList[i])+')'] = list()
    return dummyDataFrame

def getList(Obj: 'dict[str, dict]'):
    arrList = list()
    logging.info("%s", type(Obj))
    if type(Obj) is dict:
        for key in Obj:
            logging.info("%s->", key)
            arrList.extend(getList(Obj[key]))

    elif type(Obj) is list:
        logging.info("In list")
        return Obj
    
    else:
        raise "Invalid Object"

    return arrList

""" Computing line graph for component metrics"""
# import matplotlib.pylab as plt
# def createPlot(data, color, name, path: str = ""):
#     x=list()
#     for i in range(53):
#         x.append(data[i])
#     y=graphMetrics.subjectList()
#     fig = plt.figure(figsize = (20, 5))
#     plt.plot(y, x, color)
#     plt.title('Degree: '+name)
#     plt.margins(0)
#     plt.legend([name])
#     plt.savefig(
#         path+name+'.png',
#         pad_inches = 1,
#         transparent = True,
#         facecolor ="w",
#         edgecolor ='g',
#         orientation ='landscape',
#         format='png',
#         dpi=1200
#     )
#     plt.close()

# colors = ['-gD', '-rD', '-bD']
# for i in range(len(groups)):
#     createPlot(computeDegree(numpyMatrix(graphMetrics.avgFNCs[groups[i]])), colors[i], groups[i], "values/Weighted-0.15/global_values/Degree/")

# for i in range(len(groups)):
#     createPlot(computeClosenessCentrality(numpyMatrix(graphMetrics.avgFNCs[groups[i]])), colors[i], groups[i], "values/Weighted-0.15/global_values/ClosnessCentrality/")

# for i in range(len(groups)):
#     createPlot(computeParticipationCoefficient(numpyMatrix(graphMetrics.avgFNCs[groups[i]])), colors[i], groups[i], "values/Weighted-0.15/global_values/ParticipationCoefficient/")