from anidex.constants import *
from anidex.utils.common import read_yaml, create_directories
from anidex.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
    EvaluationConfig,
)
import os


class ConfigurationManager:
    def __init__(
        self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH
    ):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
            params_pooling=self.params.POOLING,
        )

        return prepare_base_model_config

    def get_training_config(self) -> TrainingConfig:
        config = self.config
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "data")
        create_directories([Path(training.root_dir)])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            mlflow_uri=config.mlflow_uri,
        )

        return training_config

    # def get_data_transformation_recommend_config(
    #     self,
    # ) -> DataTransformationRecommendConfig:
    #     config = self.config.recommed_data_preprocessing

    #     create_directories([config.root_dir])

    #     data_transformation_config = DataTransformationRecommendConfig(
    #         root_dir=config.root_dir,
    #         dataset_path=config.dataset_path,
    #         processed_dataset_path=config.processed_dataset_path,
    #         recommend_dataset_path=config.recommend_dataset_path,
    #         tracked_recommend_dataset_path=config.tracked_recommend_dataset_path,
    #     )

    #     return data_transformation_config

    def get_evaluation_config(self) -> EvaluationConfig:
        config = self.config
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "data")
        create_directories([Path(training.root_dir)])

        eval_config = EvaluationConfig(
            path_of_model="model.h5",
            training_data=Path(training_data),
            mlflow_uri="https://dagshub.com/Rajarshi12321/Anidex.mlflow",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
        )
        return eval_config
