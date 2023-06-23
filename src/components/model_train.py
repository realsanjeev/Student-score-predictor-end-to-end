import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from math import fabs

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exception import CustomException
from logger import logging
from utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    model_trainer_path=os.path.join("artifacts", "trained_model.pkl")
    models_info = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regression": KNeighborsRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Hist Gradient Boosting Regressor": HistGradientBoostingRegressor(),
                # "XGBRegressor": XGBRegressor(),
                # "CatBoosting Regressor": CatBoostRegressor(verbose=False)
                }
    models_param = {
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "Decision Tree": {
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["auto", "sqrt"]
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "Linear Regression": {},
    "K-Neighbors Regression": {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree"]
    },
    "AdaBoost Regressor": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "loss": ["linear", "square", "exponential"]
    },
    "Hist Gradient Boosting Regressor": {
        "learning_rate": [0.01, 0.1, 0.2],
        "max_iter": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "min_samples_leaf": [1, 2, 4]
    }
}


class ModelTrainer:
    def __init__(self, confidence_threshold:float =0.6):
        self.model_trainer_config = ModelTrainerConfig()
        if fabs(confidence_threshold) < 1:
            self.confidence = confidence_threshold
        else:
            logging.warning("Confidence level cannot be more than 1. Re=assgining the confidence to 0.6")
            self.confidence = 0.6

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Initiating the model training")
            models = self.model_trainer_config.models_info
            models_param = self.model_trainer_config.models_param
            model_report = evaluate_model(train_array, 
                                          test_array, 
                                          models, 
                                          models_param)

            report_df = pd.DataFrame(model_report)
            # For more readability of dataframe transpose it
            report_df = report_df.transpose()
            print(report_df)

            best_model_name = report_df['test_r2_score'].idxmax()
            best_score = report_df.loc[best_model_name, 'test_r2_score']

            if best_score < self.confidence:
                logging.critical(f"No best model is detected. Best model was: {best_model} with best score: {best_score}")
                raise CustomException(f"No best model is detected. Best model was: {best_model} with best score: {best_score}")
            # get trained model object and store for future use
            best_model = report_df.loc[best_model_name, 'model']

            save_object(
                obj=best_model,
                path=self.model_trainer_config.model_trainer_path
                )

            return best_model
        except Exception as err:
            logging.error("Problem occured while training the model")
            raise CustomException(err, sys)

if __name__=="__main__":
    print("Main initiated")