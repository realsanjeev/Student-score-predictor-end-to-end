import os
import sys
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from sklearn.model_selection import train_test_split

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
                 test_size: int=0.2, random_seed: Optional[int]=None):
        '''
        Initialize DataIngestion class.

        This class is used for data ingestion from a specified source file.

        Args:
            src_path (str): Path of the file used for data ingestion.
            test_size (float, optional): Test split ratio for splitting the dataset. Default is 0.2.
            random_seed (int, optional): Random seed used for consistent results in different computing environments
                                        when using `train_test_split`. Default is None.

        '''
        self.ingestion_data_path=DataIngestionConfig()
        self.src_path=src_path
        self.test_size=test_size
        self.random_seed=random_seed
    
    def initiate_data_ingestion(self):
        '''
        Initialize the data ingestion process.

        This method performs a test-train split on the dataset and returns the paths of the train and test splits.

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
                                                    test_size=self.test_size, random_state=self.random_seed)
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
    data_ingestion_obj = DataIngestion(src_path="project.csv")
    data_ingestion_obj.initiate_data_ingestion()