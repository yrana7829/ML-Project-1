from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig


if __name__ == "__main__":
    object = DataIngestion()
    train_data, test_data = object.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_array, test_array = data_transformation.initiate_data_transformation(train_data, test_data)
