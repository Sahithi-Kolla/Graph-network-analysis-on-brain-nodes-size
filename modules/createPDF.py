from PIL import Image
from modules.helpers import *
import os



"""Underdeveloped project"""
class PDF:

    def __init__(self):
        
        self.imagesNames = {
            "FNC": {
                "groups": [
                    "values/Weighted-0.15/FNC/groups/CN FNC.png",
                    "values/Weighted-0.15/FNC/groups/MCI FNC.png",
                    "values/Weighted-0.15/FNC/groups/AD FNC.png"
                ],
                "differences": [
                    "values/Weighted-0.15/FNC/differences/CN Vs MCI Difference FNC.png",
                    "values/Weighted-0.15/FNC/differences/CN Vs AD Difference FNC.png",
                    "values/Weighted-0.15/FNC/differences/MCI Vs AD Difference FNC.png"
                ]
            },
            "global_values": [ 
                "values/Weighted-0.15/global_values/Global Metrics.png",
            ],
            "componentMetrics": {
                "groups": [
                    "values/Weighted-0.15/componentMetrics/measures/Degree.png",
                    "values/Weighted-0.15/componentMetrics/measures/Closeness centrality.png",
                    "values/Weighted-0.15/componentMetrics/measures/Participation coefficient.png",
                ],
                "differences": [
                    "values/Weighted-0.15/componentMetrics/differences/Degree-Difference.png",
                    "values/Weighted-0.15/componentMetrics/differences/Closeness centrality-Difference.png",
                    "values/Weighted-0.15/componentMetrics/differences/Participation coefficient-Difference.png",
                ]
            },
            "crossCorrelation": {
                "groups": [
                    "values/Weighted-0.15/crossCorrelation/groups/CN: Correlation.png",
                    "values/Weighted-0.15/crossCorrelation/groups/MCI: Correlation.png",
                    "values/Weighted-0.15/crossCorrelation/groups/AD: Correlation.png"
                ],
                "differences": [
                    "values/Weighted-0.15/crossCorrelation/differences/CN-MCI: difference of correlations.png",
                    "values/Weighted-0.15/crossCorrelation/differences/CN-AD: difference of correlations.png",
                    "values/Weighted-0.15/crossCorrelation/differences/MCI-AD: difference of correlations.png"
                ]
            }
        }

        self.imagesList = list()
        for key in self.imagesNames:
            logging.info("%s->", key)
            self.imagesList.extend(getList(self.imagesNames[key]))
        
        logging.info(self.imagesList)