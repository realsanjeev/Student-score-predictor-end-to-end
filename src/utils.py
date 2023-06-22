import os
import sys
import pickle

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

        raise CustomException(err, sys)