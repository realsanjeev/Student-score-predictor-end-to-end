import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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