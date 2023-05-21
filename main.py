from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer


if __name__ == "__main__":
    object = DataIngestion()
    train_data, test_data = object.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_array, test_array = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    r2_score = model_trainer.initiate_model_training(test_array, test_array)


