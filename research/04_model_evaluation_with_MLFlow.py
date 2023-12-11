import os
from urllib.parse import urlparse

import keras
from dataclasses import dataclass
from src.cnnClassifier.constants import *
from src.cnnClassifier.utils.common import read_yaml, create_directories, save_json
import mlflow.keras

os.chdir("../")

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/avip1607/Kidney-Disease-Classification.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "avip1607"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "04dd3be8ff1e3cbe85ffd4f528883b6f9730f28a"

# Load the model
model = keras.models.load_model("artifacts/training/model.h5")


@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int


class ConfigurationManager:
    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/model.h5",
            training_data="artifacts/data_ingestion/kidney-ct-scan-image",
            mlflow_uri="https://dagshub.com/avip1607/Kidney-Disease-Classification.mlflow",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        data_generator_kwargs = dict(
            rescale=1. / 255,
            validation_split=0.30
        )

        data_flow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_data_generator = keras.preprocessing.image.ImageDataGenerator(
            **data_generator_kwargs
        )

        self.valid_generator = valid_data_generator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **data_flow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> keras.Model:
        return keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("score.json"), data=scores)

    def log_into_mlflow(self):
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_score = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_param(self.config.all_params)
            mlflow.log_metric(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )

            # Model registry does not work with file store
            if tracking_url_type_score != "file":

                # Register the model
                # There aare other way to use the Model Registry, which depends on the use case,
                # Please refer to the doc for more information:
                mlflow.keras.log_model(self.model, "model", registered_model_name="V16Model")
            else:
                mlflow.keras.log_model(self.model, "model")


try:
    config = ConfigurationManager()
    eval_config = config.get_evaluation_config()
    evaluation = Evaluation(eval_config)
    evaluation.evaluation()
    evaluation.log_into_mlflow()
except Exception as e:
    raise e
