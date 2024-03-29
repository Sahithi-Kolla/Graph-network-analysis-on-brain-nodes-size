
import pandas as pd
import seaborn as sns
import matplotlib as mt
from matplotlib import pyplot as plt
import seaborn as sns

from scipy import stats
from matplotlib import pyplot as plt
from statsmodels.stats import multitest
from copy import deepcopy

from modules.subject import GroupsInfo
from modules.voxel import VoxelCounts
from modules.graphMetrics import GraphMetrics

from modules.helpers import *

class CorrsCorrelations:
    def __init__(self, subjects: GroupsInfo, voxels: VoxelCounts, graphMetrics: GraphMetrics):
        self.subjects = subjects
        self.graphMetrics = graphMetrics
        self.voxels = voxels

        ## Get Data from above class
        self.groupsToCompute = list(self.subjects.getGroupDictionary().keys())
        self.indexList = self.subjects.getICNIndexes()
        self.domainList = self.subjects.getICNDomainNames()
        self.graphMetricGlobalMeasues = self.subjects.getGlobalGraphMetrics()
        self.graphMetricComponentMeasures = self.subjects.getComponentGraphMetrics()
        self.graphMetricsMeasures = self.graphMetricGlobalMeasues + self.graphMetricComponentMeasures 

        self.CrossListTables = dict()
        self.PListTables = dict()
        self.FDRTables = dict()
        
        self.differencePValues = dict()
        self.differenceFDRValues = dict()
        self.differenceCrossCorrelation = dict()

        self.dummyDataFrame = self.prepareDataFrame()


    def prepareDataFrame(self):
        dummyDataFrame = pd.DataFrame()
        for i in range(len(self.indexList)):
            if(i!=0 and self.domainList[i] == self.domainList[i-1]):
                dummyDataFrame['('+str(self.indexList[i])+')'] = list()
            else:
                dummyDataFrame[self.domainList[i]+'('+str(self.indexList[i])+')'] = list()
        return dummyDataFrame

    def calculateDifferencePvalues(self):
        
        order = self.groupsToCompute
        for i in range(len(order)):
            tab1 = deepcopy(self.CrossListTables[order[i]])

            columns = tab1.columns.values
            rows = tab1.index.values

            for j in range(i+1, len(order)):
                # diffPvalues = deepcopy(self.dummyDataFrame)
                # diffFDRvalues = deepcopy(self.dummyDataFrame)
                diffCrossValues = deepcopy(self.dummyDataFrame)
                
                tab2 = deepcopy(self.CrossListTables[order[j]])

                for r in rows:
                    rowValues = list()
                    for c in columns:
                        rowValues.append(tab1.at[r,c] - tab2.at[r,c])

                    diffCrossValues.loc[len(diffCrossValues.index)] = rowValues
                    # diffPvalues.loc[len(diffPvalues.index)] = rowValues
                    # diffFDRvalues.loc[len(diffFDRvalues.index)] = rowValues

                # diffPvalues.index = self.graphMetricsMeasures
                # diffFDRvalues.index = self.graphMetricsMeasures
                diffCrossValues.index = self.graphMetricsMeasures
                
                # for rowNum in range(len(rows)):
                #     graphCross = deepcopy(list(diffCrossValues.iloc[rowNum]))
                #     # print("rowname: ", rowNum)
                #     # print("diffCross: ", graphCross)
                #     for ind in range(len(graphCross)):
                #         graphCross[ind] = self.calculatePvalue( graphCross[ind], len(self.AD_Threshold.voxelCountMap[order[i]]['paths']) + len(self.AD_Threshold.voxelCountMap[order[j]]['paths']), False)

                #     diffPvalues.iloc[rowNum] = graphCross
                #     diffFDRvalues.iloc[rowNum] = multitest.fdrcorrection(graphCross, 0.05)[0]
                
                # self.generateImageFromDataFrame(diffCrossValues, order[i]+'-'+order[j]+'-diffCrossValues', vmin=-1.0, vmax=1.0)
                self.differenceCrossCorrelation[order[i]+'-'+order[j]]=diffCrossValues
                # self.generateImageFromDataFrame(diffPvalues, order[i]+'-'+order[j]+'-pvalues')
                # self.getMatrix(diffFDRvalues, diffCrossValues, order[i]+'-'+order[j]+'-fdrCross')

    def calculateStatsCrossCorrelation(self):
        for subjectName in self.groupsToCompute:

            graphMetricsTable = deepcopy(self.dummyDataFrame)
            pvaluesTables = deepcopy(self.dummyDataFrame)
            fdrTables = deepcopy(self.dummyDataFrame)

            # For Global Measure
            for graphMetricName in self.graphMetricGlobalMeasues:
                indexList = self.indexList
                corr_list_global = list()
                p_list_global = list()

                # GraphMetric List of 213-AD, 927-CN,.... subjects 
                xlist_global = self.graphMetrics.getGlobalGraphMetricValues(subjectName, graphMetricName)
                
                # 69, 53, 98, ......
                for j in range(len(indexList)):
                    
                    # VoxelCount List of 213-AD,927-CN,... for Selected Index 69,57,..... 
                    ylist_global = self.voxels.getVoxelCountsForIndex( subjectName , indexList[j])
                    # corr_list_global.append(self.AD_Threshold.calculatePearsonCrossCorrelations(xlist_global, ylist_global) )
                    prs, _ = stats.pearsonr(xlist_global, ylist_global)
                    corr_list_global.append(prs)
                    
                    size = len(ylist_global)
                    p_list_global.append(calculatePvalue(prs,size))

                graphMetricsTable.loc[len(graphMetricsTable.index)] = corr_list_global
                # pvaluesTables.loc[len(pvaluesTables.index)] = p_list_global
                # fdrTables.loc[len(fdrTables.index)] = multitest.fdrcorrection(p_list_global, 0.05)[0]

            # For Component Measure
            for graphMetricName in self.graphMetricComponentMeasures:
                indexList = self.indexList
                corr_list_component = list()
                p_list_component = list()

                # 69, 53, 98, ......
                for j in range(len(indexList)):

                    xlist_component = self.graphMetrics.getComponentGraphMetricIndValues(subjectName,graphMetricName,indexList[j])
                    ylist_component = self.voxels.getVoxelCountsForIndex(subjectName, indexList[j])
                    # corr_list_component.append(self.AD_Threshold.calculatePearsonCrossCorrelations(xlist_component, ylist_component))
                    prs, _ = stats.pearsonr(xlist_component, ylist_component)
                    corr_list_component.append(prs)
                    
                    size = len(xlist_component)
                    p_list_component.append(calculatePvalue(prs, size))
                    # p_list_component.append(round(pval, 5))

                graphMetricsTable.loc[len(graphMetricsTable.index)] = corr_list_component
                # pvaluesTables.loc[len(pvaluesTables.index)] = p_list_component
                # fdrTables.loc[len(fdrTables.index)] = multitest.fdrcorrection(p_list_component, 0.05)[0]

            graphMetricsTable.index = self.graphMetricsMeasures
            # pvaluesTables.index = self.graphMetricsMeasures
            # fdrTables.index = self.graphMetricsMeasures

            self.CrossListTables[subjectName] = graphMetricsTable
            # self.PListTables[subjectName] = pvaluesTables
            # self.FDRTables[subjectName] = fdrTables

    def savePValues(self):
        for key in self.PListTables:
            Ptable = self.PListTables[key]
            # CrossCorrTable = self.CrossListTables[key]
            # Ptable.to_csv(key+'-p_values.csv')
            self.generateImageFromDataFrame(Ptable, key+'-pTable')

    def saveCrossCorrelation(self):
        vmin = 100
        vmax = -100
        for key in self.CrossListTables:
            vmin = min(vmin, min(self.CrossListTables[key].min().values))
            vmax = max( vmax, max(self.CrossListTables[key].max().values) )
        print(vmin, vmax)
        for key in self.CrossListTables:
            CrossCorr = self.CrossListTables[key]
            # CrossCorrTable = self.CrossListTables[key]
            # Ptable.to_csv(key+'-p_values.csv')
            self.generateImageFromDataFrame(CrossCorr, name=key+": Cross Correlation", vmin=vmin, vmax=vmax)
    
    def saveDifferenceCrossCorrelation(self):
        vmin = 100
        vmax = -100
        for key in self.differenceCrossCorrelation:
            vmin = min( vmin, min(self.differenceCrossCorrelation[key].min().values))
            vmax = max( vmax, max(self.differenceCrossCorrelation[key].max().values) )
        print(vmin, vmax)
        for key in self.differenceCrossCorrelation:
            diffCrossCorr = self.differenceCrossCorrelation[key]
            # CrossCorrTable = self.CrossListTables[key]
            # Ptable.to_csv(key+'-p_values.csv')
            self.generateImageFromDataFrame(diffCrossCorr, name=key+": Difference Correlations", vmin=vmin, vmax=vmax)
            

    def saveFDRValues(self):
        for key in self.FDRTables:
            FDRtable = self.FDRTables[key]
            # graphMetricValues = self.CrossListTables[key]
            # self.getMatrix(deepcopy(FDRtable), deepcopy(graphMetricValues), key)
            # FDRtable.to_csv(key+'-fdr_values.csv')
            self.generateImageFromDataFrame(FDRtable, key+'-fdrTable')

    def getMatrix(self, table, cross, subjectName):
        diff = pd.DataFrame().reindex_like(table)

        columns = table.columns.values
        rows = cross.index.values

        for i in rows:
            for j in columns:
                # diff.at[i,j] = table.at[i,j] * cross.at[i,j]
                diff.at[i,j] = 1 if table.at[i,j] else 0

        self.generateImageFromDataFrame(deepcopy(diff), subjectName, 0, 1)

    def generateImageFromDataFrame(self, table, name, vmin=None, vmax=None):
        fig, ax = mt.pyplot.subplots(figsize=(90,10))
        columnsName = table.columns.values
        rowsName = table.index.values
        # if vmin != None and vmax != None:
        sd = sns.heatmap(
                table, cmap='coolwarm', square=True, ax=ax, linewidth='0.8', center=0, vmin=vmin,
                xticklabels=columnsName, yticklabels=rowsName,  annot=True, vmax=vmax, annot_kws = { "size": 18 },
                cbar_kws={"fraction": 0.046, "pad": 0.01}
            )
        # else:
        #     sd = sns.heatmap(
        #             table, cmap='coolwarm', square=True, ax=ax, linewidth='0.5', center=0, vmin=-0.5,
        #             xticklabels=columnsName, yticklabels=rowsName,  annot=True, vmax=0.5,
        #             cbar_kws={"fraction": 0.046, "pad": 0.04}
        #         )

        sd.set_yticklabels(labels= rowsName,fontsize=30, weight='bold')
        sd.set_xticklabels(labels= columnsName,fontsize=30, weight='bold')
        plt.margins(0)
        ax.set_title(name, fontsize=40,  fontweight='bold')
        # ax.tick_params(axis='both', which='major', labelsize=30,width=3, color=(0,0,0))
        # ax.tick_params(axis='both', which='minor', labelsize=30,width=3, color=(0,0,0))
        plt.savefig(name+'.png')