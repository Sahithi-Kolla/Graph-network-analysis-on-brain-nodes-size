from modules.subject import GroupsInfo
from modules.helpers import *

from nilearn.image import index_img
from copy import deepcopy

class VoxelCounts:
    " \
        datasetPath : 4D images path \
        componentsPath: indexes path \
        subjectList: List of subjects to calculate ex: AD, CN, ....  \
    "
    voxels = None

    def __init__(self):
        self.subjectsData = GroupsInfo.getInstance()
        self.subjectsToCalculate = self.subjectsData.getGroupDictionary().keys()
        self.voxelCountMap = dict()
        self.prepareVoxelCountMap()

    def prepareVoxelCountMap(self):
        " \
            iterate over the list of Subjects like AD, CN, ....     \
        "
        for subjectName in self.subjectsToCalculate:
            self.voxelCountMap[subjectName]=dict()
            self.voxelCountMap[subjectName]["paths"] = self.subjectsData.getGroupDatasetPaths(subjectName)
            self.voxelCountMap[subjectName]["indexes"] = dict()
            for i in self.subjectsData.getICNIndexes():
                self.voxelCountMap[subjectName]["indexes"][i]=list()

    def calculateVoxelCount(self):
        for subjectName in self.subjectsToCalculate:
            for path in self.voxelCountMap[subjectName]["paths"]:
                spacialMapName = path + self.subjectsData.getImageFileName()
                spacialMap = load4DImageData(spacialMapName)
                for index in self.voxelCountMap[subjectName]["indexes"]:
                    actualIndex=index-1
                    componentImg = index_img(spacialMap, actualIndex)
                    componentImgData = componentImg.get_fdata()
                    voxelCount = countVoxels(componentImgData)
                    self.voxelCountMap[subjectName]["indexes"][index].append(voxelCount)
    ####################
    # Getter Functions #
    ####################

    @classmethod
    def getInstance(cls):
        if cls.voxels == None:
            cls.voxels = cls()
        return cls.voxels

    """\
        {
            '69': [ v1, v2, v3, ...... totalSubjects ],
            '54': [ v1, v2, v3, .........],
            ....
            ....
        }
    """
    def getVoxelCounts(self, subjectName):
        return deepcopy(self.voxelCountMap[subjectName]["indexes"])
    
    """
        [ v1, v2, v3, v4, ..........totalSubjects]
    """
    def getVoxelCountsForIndex(self, subjectName, index): 
        return deepcopy(self.voxelCountMap[subjectName]["indexes"][index])