# this notebook will be used to import data from the data lake and prepare it for training
# we will use local data here for simplicity however it real situation we would use your lake house.

from dataclasses import dataclass
import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer 

import pandas as pd

from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    """
    Data Ingestion Config
    """

    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Initiates Data Ingestion
        """
        logging.info("Initiating Data Ingestion")
        try:
            # this can be replaced with your lake house delta tables
            df = pd.read_csv("notebooks/data/stud.csv")
            logging.info("reading raw data shape: {}".format(df.shape))

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Data Splitting Started")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )
            logging.info("Data ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            logging.error(f"Error occurred while initiating data ingestion: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    training_path, test_path = data_ingestion.initiate_data_ingestion()
    logging.info("initializing Data Transformation object")
    data_transformation = DataTransformation()
    logging.info("initiating data transformation")
    train_array, test_array, preprocessor_file_path = data_transformation.initiate_data_transformation(training_path, test_path)
    logging.info("data transformation completed")
    logging.info("initializing Model Trainer object")
    mode_trainer = ModelTrainer()
    logging.info("initiating model training")
    mode_trainer.initiate_model_trainer(train_array, test_array)
    logging.info("model training completed")
    
