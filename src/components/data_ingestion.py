import os
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass



# Initiating the config for this class:- inputs 
@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifact', 'train.csv')
    test_data_path:str=os.path.join('artifact', 'test.csv')
    raw_data_path:str=os.path.join('artifact', 'data.csv')

# Define the main class for data ingestion

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")

        try:
            df = pd.read_csv('notebooks/data/data.csv')
            logging.info(" Data is read as dataframe")

            # Make directory to save the data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            logging.info("train test split started")

            # Make the train test split and save them to directories
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            train_set, test_set = train_test_split(df, test_size=0.2)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Train test split completed, data is saved into respective directories")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
            