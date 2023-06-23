import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exception import CustomException
from logger import logging
from utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path=os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.preprocessor_config = DataTransformationConfig()
    
    def get_transformation_obj(self, num_features, categorical_features):
        '''
        Returns the preprocessor object
        '''
        numerical_pipeline = Pipeline(steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
        categorical_pipeline = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoding", OneHotEncoder())
        ])

        try:
            preprocessor = ColumnTransformer([
                ("numerical_pipeline", numerical_pipeline, num_features),
                ("categorical_pipeline", categorical_pipeline, categorical_features)]
            )
            
            return preprocessor
        except Exception as err:
            logging.error("Error occur when constructing preprocess transformer")
            raise CustomException(err, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        '''
        Initialize data transformation for features
        Args:
            `train_path`: path(str) -> path for training data set
            `test_path`: path(str) -> paath for testing data set
        Returns:
            `train_arr`: np.array() -> array of training data after preprocessing feature
            `test_arr`: np.array() -> array of test data after transforming feature
            `preprocessing_path`: saved path of preprocessing object

        '''
        try:
            train_data_df = pd.read_csv(train_path)
            test_data_df = pd.read_csv(test_path)
            logging.info("Reading training and test dataset from path")

            numerical_columns = [
                "writing_score",
                "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            # initiating preprocessing object
            preprocessing_obj = self.get_transformation_obj(numerical_columns, categorical_columns)

            traget_column="math_score"

            # separatiom of record in input and output features in dataframe
            input_feature_train_data = train_data_df.drop(columns=[traget_column], axis=1)
            target_feature_train_data = train_data_df[traget_column]

            # separating data in input and target featue in test_data_df
            input_feature_test_data = test_data_df.drop(columns=[traget_column], axis=1)
            target_feature_test_data = test_data_df[traget_column]

            logging.info("Separating feature data and target data from train and test data set")

            # fit data for standrandize data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_data)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_data)

            train_arr = np.c_[
                input_feature_train_arr, 
                np.array(target_feature_train_data)
            ]
            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_data)
            ]
            save_object(
                obj=preprocessing_obj,
                path=self.preprocessor_config.preprocessor_obj_path
            )

            return (
                train_arr,
                test_arr,
                self.preprocessor_config.preprocessor_obj_path
            )
        except Exception as e:
            raise CustomException(e, sys)