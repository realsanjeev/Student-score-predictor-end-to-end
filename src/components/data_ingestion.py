import os
import sys
import argparse
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from sklearn.model_selection import train_test_split
from data_transformation import DataTransformation
from model_train import ModelTrainer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exception import CustomException
from logger import logging


@dataclass
class DataIngestionConfig:
    raw_data_path: str= os.path.join("artifacts", "raw.csv")
    train_data_path: str= os.path.join("artifacts", "train.csv")
    test_data_path: str= os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self, src_path: str=None, 
                 test_size: float=0.2, random_seed: Optional[int]=None):
        '''
        Initialize DataIngestion class.

        This class is used for data ingestion from a specified source file.

        Args:
            src_path (str): Path of the file used for data ingestion.
            test_size (float, optional): Test split ratio for splitting the dataset. Default is 0.2.
            random_seed (int, optional): Random seed used for consistent results \n
            in different computing environments\n
            when using `train_test_split`. Default is None.

        '''
        self.ingestion_data_path=DataIngestionConfig()
        self.src_path=src_path
        self.test_size=test_size
        self.random_seed=random_seed
    
    def initiate_data_ingestion(self) -> tuple:
        '''
        Initialize the data ingestion process.

        This method performs a test-train split on the dataset \n
        and returns the paths of the train and test splits.

        Returns:
            train_path (str): Path of the train split.
            test_path (str): Path of the test split.

        '''
        logging.info("Initiating data ingestion method in DataIngestion class")
        try:
            df = pd.read_csv(self.src_path)
            logging.info("Reading file from csv as DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_data_path.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_data_path.raw_data_path,
                      header=True, index=False)
            
            logging.info("Train test split initiated")
            train_set, test_set =  train_test_split(df, 
                                                    test_size=self.test_size, 
                                                    random_state=self.random_seed)
            train_set.to_csv(self.ingestion_data_path.train_data_path, 
                             header=True, index=False)
            test_set.to_csv(self.ingestion_data_path.test_data_path,
                            header=True, index=False)
            return (
                self.ingestion_data_path.train_data_path,
                self.ingestion_data_path.test_data_path
            )
        except Exception as err:
            logging.error(err)
            raise CustomException(err, sys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="[INFO] Train-test-split for dataset")
    parser.add_argument('-s', '--split', 
                        help="Train-test-split for dividing dataset for training", type=float)
    args = parser.parse_args()
    
    if args.split is None:
        test_train_split = 0.2  # Set a default value if split is not provided
        logging.warning('Split value not provided. Using default split value of 0.2')
    else:
        test_train_split = abs(args.split)
        if test_train_split > 0.5:
            test_train_split = 0.2
            logging.warning('Train test split cannot be more than 0.5')

    data_ingestion_obj = DataIngestion(src_path="project.csv", test_size=test_train_split)
    train_path, test_path = data_ingestion_obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ =data_transformation.initiate_data_transformation(train_path, test_path)
    model_trainer = ModelTrainer()
    report = model_trainer.initiate_model_training(train_arr, test_arr)