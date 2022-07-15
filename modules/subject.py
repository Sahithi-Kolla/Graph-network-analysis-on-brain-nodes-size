from modules.helpers import *

class GroupsInfo:
    subjectsData = None

    def __init__(self, datasetPath, componentsPath):
        print("++Initiating GroupInfo Class....")
        self.__brainImageFileName = "adni_aa__sub01_component_ica_s1_.nii"
        self.__timeCourseFileName = "adni_aa__sub01_timecourses_ica_s1_.nii"
        self.__FNCmatFileName = "adni_aa__postprocess_results.mat"
        self.__component_key = "fnc_corrs_all"
        
        self.__groupDatasetPaths = dict()

        self.__graphMetricGlobalMeasues = [
            # Global Measures
            "Global efficiency",
            "Characteristic path length",
            "Clustering coefficient",
        ]

        self.__graphMetricComponentMeasures = [
            # Component Measures
            "Degree",
            "Closeness centrality",
            "Participation coefficient"
        ]

        self.__groupDictionary = {
            "CN": "Controls",
            "MCI": "Mild Cognitive Impairment",
            "AD": "Alzhiemeris",
        }

        self.__subjectsData = readFromCSV(datasetPath)
        self.__lengthOfAllSubjects = len(self.__subjectsData.index)
        self.__componentData = readFromCSV(componentsPath)

        self.__domainList = list(self.__componentData["icn_domain"])
        self.__indexList = list(self.__componentData["icn_index"])

        self.__modifySubjectsPath()
        self.__getAllGroupsDatasetPaths()
        print("--Intiated GroupInfo Class....")

    def __modifySubjectsPath(self):
        for i in range(self.__lengthOfAllSubjects):
            self.__subjectsData.at[i,"fc_dir"] = self.__subjectsData.iloc[i]["fc_dir"].replace("FC","GIGICA")
    
    def __getAllGroupsDatasetPaths(self):
        for groupName in self.__groupDictionary.keys():
            self.__groupDatasetPaths[groupName] = list()

        for i in range(self.__lengthOfAllSubjects):
            groupName = self.__subjectsData.at[i, "ResearchGroup"]
            if groupName in self.__groupDictionary:
                self.__groupDatasetPaths[groupName].append(self.__subjectsData.at[i,"fc_dir"])
    
    ####################
    # Getter Functions #
    ####################

    @classmethod
    def getInstance(cls):
        # pass
        # print("Hello")
        if cls.subjectsData == None:
            cls.subjectsData = cls('ADNI_demos.txt', 'NM_icns_info.csv')
        return cls.subjectsData

    def getGroupDatasetPaths(self, groupName):
        if groupName in self.__groupDatasetPaths:
            return self.__groupDatasetPaths[groupName]
        return None
    
    def getGroupDataSetSize(self, groupName=None):
        if groupName in self.__groupDatasetPaths:
            return len(self.__groupDatasetPaths[groupName])
        else:
            totalDatasets = 0
            for groupName in self.__groupDictionary.keys():
                totalDatasets += self.__groupDatasetPaths[groupName]
            return totalDatasets

    def getImageFileName(self):
        return self.__brainImageFileName
    
    def getTimecourseFileName(self):
        return self.__timeCourseFileName

    def getFNCFileName(self):
        return self.__FNCmatFileName
    
    def getComponentKey(self):
        return self.__component_key
    
    def getGlobalGraphMetrics(self):
        return self.__graphMetricGlobalMeasues
    
    def getComponentGraphMetrics(self):
        return self.__graphMetricComponentMeasures

    def getGroupDictionary(self):
        return self.__groupDictionary
    
    def getICNDomainNames(self):
        return self.__domainList
    
    def getICNIndexes(self):
        return self.__indexList