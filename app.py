import traceback
from src.Insurance_prediction.logger import logging
from src.Insurance_prediction.exception import CustomException
import sys
from src.Insurance_prediction.components.data_ingestion import DataIngestion
from src.Insurance_prediction.components.data_ingestion import DataIngestionConfig
from src.Insurance_prediction.components.data_transformation import DataTransformationConfig, DataTransformation
from Insurance_prediction.components.model_trainer import ModelTrainerConfig, ModelTrainer



if __name__ == "__main__":
    logging.info("The execution has started")


    try:
       data_ingestion=DataIngestion()
       train_data_path, test_data_path=data_ingestion.initiate_data_ingestion()
        
       data_transformation=DataTransformation()
       train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

       model_trainer=ModelTrainer()
       print(model_trainer.initiate_model_trainer(train_arr,test_arr))

    except Exception as e:
         exc_type, exc_value, exc_tb = sys.exc_info()
         tb = traceback.format_exception(exc_type, exc_value, exc_tb)
         error_message = ''.join(tb)
    raise CustomException(error_message)
      
        