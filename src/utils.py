import os
import sys
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from exception import CustomException
from logger import logging

def save_object(obj, path):
    try:
        with open(path, "wb") as file_p:
            pickle.dump(obj, file_p, pickle.HIGHEST_PROTOCOL)
            logging.info(f"Saving object in path: {path}")
    except Exception as err:
        logging.error(f"Error Saving the object: {err}")
        raise CustomException(err, sys)
    
def load_object(path):
    try:
        with open(path, "rb") as file_p:
            byte_stream =  pickle.load(path)
        return byte_stream
    except Exception as err:
        logging.error("Error while loading the file in path: {path}")
        raise CustomException(err, sys)
    
def evaluate_model(train_array: tuple, 
                    test_array: tuple, 
                    models: dict, 
                    params_grid: dict) -> dict:
    """
    Returns report for all model used to train
    
    Args:
        train_array: np.array()
        test_array: np.array()
        models: dict
        params_grid: dict
    Result:
        report: dict -> report of model summary
    """
    # unpacking the tuple to feature and target
    X_train, y_train = (
        train_array[:,:-1],
        train_array[:, -1]
    )
    X_test, y_test = (
        test_array[:,:-1],
        test_array[:, -1]
    )
    try:
        report = {}
        for key in models.keys():
            print("*"*9, key, "*"*3)
            estimator = models[key]
            param = params_grid.get(key, 0)
            try:
                model = GridSearchCV(estimator, param, cv=5)
            except Exception as err:
                logging.warn(f"Problem while training with model without providing params: {estimator}. Training with default parameter")
                logging.critical(f"{err}")
                model = estimator
            # fitting training data in the model
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            score = {"train_r2_score": train_model_score,
                     "test_r2_score": test_model_score}
            report[key] = score

        return report
    except Exception as err:
        logging.error("Error while evaluating model")
        raise CustomException(err, sys)