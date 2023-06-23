import os
import sys

from src.utils import load_object
from src.logger import logging
from src.exception import CustomException

class PredictPipeline:
    def __init__(self) -> None:
        self.features = {
            "categorical": {
                "gender":['male', 'female'],
                "race_ethnicity":['group A', 'group B', 'group C', 'group D'],
                "parental_level_of_education":["bachelor's degree", "some college", 
                                               "master's degree", "associate's degree"],
                "lunch":["standard", "free/reduced"],
                "test_preparation_course":["completed", "none"],
            },
            "numerical": [
                   "writing_score",
                    "reading_score"
                    ]
        }
        self.preprocess_obj_path = os.path.join("artifacts", "preprocessor.pkl")
        self.model_obj = os.path.join("artifacts", "trained_model.pkl")

    def preprocess(self, gender: str,
                    race_ethnicity: str,
                    parental_level_of_education: str,
                    lunch: str,
                    test_preparation_course: str,
                    reading_score: int,
                    writing_score: int):
        logging.info("Preprocessing of user data started")
        freature_data = [gender, race_ethnicity, parental_level_of_education, 
                         lunch, test_preparation_course, reading_score, writing_score]
        try:
            preprocess_obj = load_object(self.preprocess_obj_path)
            scaled_data = preprocess_obj.transform(freature_data)
            return scaled_data
        except Exception as err:
            logging.error(f"Error occurs while working with user data: '{err}' ")
            raise CustomException(err, sys)

    def get_features(self):
        return self.features 
    
    def predict_score(self, scaled_data):
        logging.info(f"Predction started with preprocessed data")
        try:
            model_obj = load_object(self.model_obj)
            model_obj.fit(scaled_data)
            return model_obj
        except Exception as err:
            logging.error(f"Error while predicting result. {err}")
            raise CustomException(err, sys)