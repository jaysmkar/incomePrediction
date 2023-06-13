import os, sys 
from src.logger import logging 
from src.exception import CustomException
from src.component.data_ingestion import DataIngestion
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTrainer
from dataclasses import dataclass 

if __name__ == "__main__":
    obj = DataIngestion() 
    train_data_path, test_data_path = obj.inititate_data_ingestion()
    
    data_transformation = DataTransformation() 
    train_array, test_array, _ = data_transformation.inititate_data_transformation(train_data_path, test_data_path)

    model_trainer = ModelTrainer()
    model_trainer.inititate_model_trainer(train_array, test_array)
