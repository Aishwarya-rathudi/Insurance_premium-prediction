from src.Insurance_prediction.logger import logging
from src.Insurance_prediction.exception import CustomException
import sys
from src.Insurance_prediction.components.data_ingestion import DataIngestion
from src.Insurance_prediction.components.data_ingestion import DataIngestionConfig



if __name__ == "__main__":
    logging.info("The execution has started")


    try:
       data_ingestion=DataIngestion()
       data_ingestion.initiate_data_ingestion()
    except Exception as e:
        logging.info("custom exception")
        raise CustomException(e,sys)