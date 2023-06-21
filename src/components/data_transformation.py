import os
import sys
import numpy as np
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path=os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        pass

    def initiate_data_transformation(self):
        pass

if __name__=="__main__":
    data_obj = DataTransformation()