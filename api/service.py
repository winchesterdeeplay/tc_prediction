import time
import logging
from typing import Any
from pathlib import Path

from inference import ONNXInferencePipeline, ONNXInferencePipelineFactory
from .models import CycloneData, Prediction, cyclone_data_to_dataframe, dataframe_to_predictions, ModelInfoResponse


class CyclonePredictionService:
    """Сервис для предсказания траекторий циклонов."""

    def __init__(self, model_path: str, pipeline_type: str = "fast", sequence_config: dict | None = None):
        """
        Инициализация сервиса предсказания.

        Parameters
        ----------
        model_path : str
            Путь к ONNX модели
        pipeline_type : str
            Тип пайплайна: "fast", "memory", "gpu"
        sequence_config : dict, optional
            Конфигурация последовательностей
        """
        self.model_path = Path(model_path)
        self.pipeline_type = pipeline_type
        self.sequence_config = sequence_config or {"min_history_length": 1, "max_history_length": 50}

        self._pipeline: ONNXInferencePipeline | None = None
        self._load_pipeline()

        logging.info(f"✅ Сервис предсказания инициализирован с моделью: {model_path}")

    def _load_pipeline(self) -> None:
        """Загружает пайплайн в зависимости от типа."""
        try:
            if self.pipeline_type == "fast":
                self._pipeline = ONNXInferencePipelineFactory.create_fast_inference(
                    str(self.model_path), self.sequence_config
                )
            elif self.pipeline_type == "memory":
                self._pipeline = ONNXInferencePipelineFactory.create_memory_efficient(
                    str(self.model_path), self.sequence_config
                )
            elif self.pipeline_type == "gpu":
                self._pipeline = ONNXInferencePipelineFactory.create_gpu_inference(
                    str(self.model_path), self.sequence_config
                )
            else:
                self._pipeline = ONNXInferencePipeline(str(self.model_path), sequence_config=self.sequence_config)

            logging.info(f"✅ Пайплайн типа '{self.pipeline_type}' загружен успешно")

        except Exception as e:
            logging.error(f"❌ Ошибка загрузки пайплайна: {e}")
            raise RuntimeError(f"Не удалось загрузить пайплайн: {e}")

    def predict(self, cyclone_data: list[CycloneData], horizon_hours: int, batch_size: int = 256) -> dict[str, Any]:
        """
        Делает предсказание для списка циклонов.

        Parameters
        ----------
        cyclone_data : list[CycloneData]
            Список данных о циклонах
        horizon_hours : int
            Горизонт прогноза в часах
        batch_size : int
            Размер батча для инференса

        Returns
        -------
        dict[str, Any]
            Результат предсказания с метаданными
        """
        if self._pipeline is None:
            raise RuntimeError("Пайплайн не загружен")

        start_time = time.time()

        try:
            # Конвертация в DataFrame
            df = cyclone_data_to_dataframe(cyclone_data)

            # Предсказание
            predictions_df = self._pipeline.predict(df=df, horizon_hours=horizon_hours, batch_size=batch_size)

            # Конвертация результатов
            predictions_dict = dataframe_to_predictions(predictions_df)
            predictions = [Prediction(**pred) for pred in predictions_dict]

            processing_time = (time.time() - start_time) * 1000  # в миллисекундах

            return {
                "predictions": predictions,
                "horizon_hours": horizon_hours,
                "total_predictions": len(predictions),
                "processing_time_ms": round(processing_time, 2),
            }

        except Exception as e:
            logging.error(f"❌ Ошибка предсказания: {e}")
            raise RuntimeError(f"Ошибка предсказания: {e}")

    def predict_multiple_horizons(
        self, cyclone_data: list[CycloneData], horizons: list[int] = [6, 12, 24, 48], batch_size: int = 256
    ) -> dict[str, Any]:
        """
        Делает предсказания для нескольких горизонтов.

        Parameters
        ----------
        cyclone_data : list[CycloneData]
            Список данных о циклонах
        horizons : list[int]
            Список горизонтов прогноза
        batch_size : int
            Размер батча для инференса

        Returns
        -------
        dict[str, Any]
            Результаты предсказаний по горизонтам
        """
        if self._pipeline is None:
            raise RuntimeError("Пайплайн не загружен")

        start_time = time.time()

        try:
            # Конвертация в DataFrame
            df = cyclone_data_to_dataframe(cyclone_data)

            # Предсказания для всех горизонтов
            results = self._pipeline.predict_multiple_horizons(df=df, horizons=horizons, batch_size=batch_size)

            # Обработка результатов
            predictions_by_horizon = {}
            total_predictions = 0

            for horizon, result_df in results.items():
                if result_df is not None:
                    predictions_dict = dataframe_to_predictions(result_df)
                    predictions = [Prediction(**pred) for pred in predictions_dict]
                    predictions_by_horizon[str(horizon)] = predictions
                    total_predictions += len(predictions)
                else:
                    predictions_by_horizon[str(horizon)] = []

            processing_time = (time.time() - start_time) * 1000  # в миллисекундах

            return {
                "predictions": predictions_by_horizon,
                "horizons": horizons,
                "total_predictions": total_predictions,
                "processing_time_ms": round(processing_time, 2),
            }

        except Exception as e:
            logging.error(f"❌ Ошибка предсказания для нескольких горизонтов: {e}")
            raise RuntimeError(f"Ошибка предсказания для нескольких горизонтов: {e}")

    def get_model_info(self) -> ModelInfoResponse:
        """
        Возвращает информацию о модели.

        Returns
        -------
        ModelInfoResponse
            Информация о модели
        """
        if self._pipeline is None:
            raise RuntimeError("Пайплайн не загружен")

        try:
            model_info = self._pipeline.get_model_info()

            # Ensure proper typing for the model info values
            inputs = model_info.get("inputs", [])
            if not isinstance(inputs, list):
                inputs = []
            # Ensure all elements are strings
            inputs = [str(x) for x in inputs]

            outputs = model_info.get("outputs", [])
            if not isinstance(outputs, list):
                outputs = []
            # Ensure all elements are strings
            outputs = [str(x) for x in outputs]

            providers = model_info.get("providers", [])
            if not isinstance(providers, list):
                providers = []
            # Ensure all elements are strings
            providers = [str(x) for x in providers]

            feature_dimensions = model_info.get("feature_dimensions", {})
            if not isinstance(feature_dimensions, dict):
                feature_dimensions = {}
            # Ensure all values are integers
            feature_dimensions = {str(k): int(v) for k, v in feature_dimensions.items()}

            return ModelInfoResponse(
                model_path=str(self.model_path),
                inputs=inputs,
                outputs=outputs,
                providers=providers,
                feature_dimensions=feature_dimensions,
            )

        except Exception as e:
            logging.error(f"❌ Ошибка получения информации о модели: {e}")
            raise RuntimeError(f"Ошибка получения информации о модели: {e}")

    def is_healthy(self) -> bool:
        """
        Проверяет здоровье сервиса.

        Returns
        -------
        bool
            True если сервис здоров
        """
        return self._pipeline is not None and self.model_path.exists()

    def reload_model(self) -> None:
        """Перезагружает модель."""
        logging.info("🔄 Перезагрузка модели...")
        self._load_pipeline()
        logging.info("✅ Модель перезагружена успешно")
