import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from core.features import FeatureConfig
from data_processing import InferenceDataProcessor
from data_processing.dataset_models import SequenceConfig


class ONNXInferencePipeline:
    def __init__(
        self,
        onnx_model_path: str,
        sequence_config: dict | None = None,
        providers: list[str] | None = None,
        session_options: ort.SessionOptions | None = None,
        validate_data: bool = True,
    ):
        """
        Инициализация ONNX Inference Pipeline.

        Parameters
        ----------
        onnx_model_path : str
            Путь к ONNX файлу модели
        sequence_config : dict, optional
            Конфигурация для создания последовательностей
        providers : list[str], optional
            Список провайдеров для ONNX Runtime (например, ['CPUExecutionProvider', 'CUDAExecutionProvider'])
        session_options : ort.SessionOptions, optional
            Дополнительные опции для ONNX Runtime сессии
        max_storm_history : int, optional
            Максимальное количество записей для каждого шторма в памяти (по умолчанию 100)
        """
        self.onnx_model_path = Path(onnx_model_path)

        if not self.onnx_model_path.exists():
            raise FileNotFoundError(f"ONNX модель не найдена: {onnx_model_path}")

        # Настройка провайдеров
        if providers is None:
            providers = ["CPUExecutionProvider"]

        # Настройка опций сессии
        if session_options is None:
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = 1
            session_options.inter_op_num_threads = 1

        # Загрузка ONNX модели
        try:
            self.ort_session = ort.InferenceSession(
                str(self.onnx_model_path), sess_options=session_options, providers=providers
            )
            logging.info(f"✅ ONNX модель загружена: {onnx_model_path}")
            logging.info(f"   Провайдеры: {self.ort_session.get_providers()}")
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки ONNX модели: {e}")

        if isinstance(sequence_config, dict):
            seq_config = SequenceConfig(**sequence_config)
        else:
            seq_config = sequence_config
            
        self.data_processor = InferenceDataProcessor(
            sequence_config=seq_config,
            validate_data=validate_data,
        )

        # Получение информации о модели
        self._model_info = self._get_model_info()
        logging.info(f"   Входы: {self._model_info['inputs']}")
        logging.info(f"   Выходы: {self._model_info['outputs']}")

        # Конфигурация фич
        self.feature_cfg = FeatureConfig()
        self.feature_dims = self.feature_cfg.get_feature_dimensions()

    def _get_model_info(self) -> dict[str, list[str]]:
        """Получает информацию о входных и выходных узлах модели."""
        inputs = [input.name for input in self.ort_session.get_inputs()]
        outputs = [output.name for output in self.ort_session.get_outputs()]
        return {"inputs": inputs, "outputs": outputs}

    def predict(self, df: pd.DataFrame, horizon_hours: int = 24, batch_size: int = 256) -> pd.DataFrame:
        """
        Делает предсказание для сырых данных циклонов.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame с сырыми данными циклонов
        horizon_hours : int
            Горизонт прогноза в часах (6, 12, 24, 48)
        batch_size : int
            Размер батча для инференса

        Returns
        -------
        pd.DataFrame
            DataFrame с предсказаниями (dlat, dlon) и исходными координатами
        """
        # Проверяем, что горизонт поддерживается (стандартные горизонты)
        supported_horizons = [6, 12, 24, 48]
        if horizon_hours not in supported_horizons:
            raise ValueError(f"Horizon {horizon_hours} not supported. Supported horizons: {supported_horizons}")

        # Создаем датасет для инференса
        inference_dataset = self.data_processor.build_dataset(df)

        # Делаем предсказания через ONNX
        predictions = self._predict_with_onnx(inference_dataset.X, batch_size=batch_size)

        # Добавляем предсказания к результату
        result_df = inference_dataset.X.copy()
        result_df["dlat_pred"] = predictions[:, 0]
        result_df["dlon_pred"] = predictions[:, 1]

        # Исходные координаты уже есть в result_df благодаря улучшенному DataProcessor
        # Проверяем их наличие
        if "lat_deg" not in result_df.columns or "lon_deg" not in result_df.columns:
            raise ValueError("DataProcessor должен возвращать lat_deg и lon_deg в inference_dataset")

        # Вычисляем предсказанные координаты
        result_df["lat_pred"] = result_df["lat_deg"] + result_df["dlat_pred"]
        result_df["lon_pred"] = result_df["lon_deg"] + result_df["dlon_pred"]

        return result_df

    def predict_multiple_horizons(
        self, df: pd.DataFrame, horizons: list[int] = [6, 12, 24, 48], batch_size: int = 256
    ) -> dict[int, pd.DataFrame | None]:
        """
        Делает предсказания для нескольких горизонтов.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame с сырыми данными циклонов
        horizons : list[int]
            Список горизонтов прогноза в часах
        batch_size : int
            Размер батча для инференса

        Returns
        -------
        dict[int, pd.DataFrame | None]
            Словарь с предсказаниями для каждого горизонта
        """
        results = {}

        for horizon in horizons:
            try:
                results[horizon] = self.predict(df, horizon, batch_size)
                logging.info(f"✅ Предсказание для горизонта {horizon}h завершено")
            except Exception as e:
                logging.error(f"❌ Ошибка предсказания для горизонта {horizon}h: {e}")
                results[horizon] = None

        return results

    def _predict_with_onnx(self, X: pd.DataFrame, batch_size: int = 256) -> np.ndarray:
        """
        Делает предсказания используя ONNX модель.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame с подготовленными данными
        batch_size : int
            Размер батча для инференса

        Returns
        -------
        np.ndarray
            Предсказания [batch_size, 2] (dlat, dlon)
        """
        # Валидация входных данных
        self._validate_input(X)

        # Извлекаем последовательности и статические фичи
        sequences = [np.array(seq, dtype=np.float32) for seq in X[self.feature_cfg.sequences_column].tolist()]
        static_cols = self.feature_cfg.static_features
        static_features = X[static_cols].values.astype(np.float32)

        all_predictions = []

        # Обрабатываем батчами
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i : i + batch_size]
            batch_static = static_features[i : i + batch_size]

            # Паддинг для батча
            max_len = max(seq.shape[0] for seq in batch_sequences)
            batch_size_actual = len(batch_sequences)
            feature_dim = batch_sequences[0].shape[1]

            padded_batch = np.zeros((batch_size_actual, max_len, feature_dim), dtype=np.float32)
            seq_lengths = np.zeros(batch_size_actual, dtype=np.int64)

            for j, seq in enumerate(batch_sequences):
                seq_len = seq.shape[0]
                padded_batch[j, :seq_len] = seq
                seq_lengths[j] = seq_len

            # ONNX предсказания
            try:
                inputs = {"sequences": padded_batch, "static_features": batch_static, "seq_lengths": seq_lengths}

                outputs = self.ort_session.run(None, inputs)
                batch_predictions = outputs[0]
                all_predictions.append(batch_predictions)

            except Exception as e:
                logging.error(f"❌ Ошибка ONNX предсказания: {e}")
                logging.error(
                    f"   Размеры: sequences={padded_batch.shape}, static={batch_static.shape}, lengths={seq_lengths.shape}"
                )
                raise

        return np.vstack(all_predictions)

    def _validate_input(self, X: pd.DataFrame) -> None:
        """
        Валидирует входные данные.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame для валидации
        """
        # Проверяем наличие колонки с последовательностями
        if self.feature_cfg.sequences_column not in X.columns:
            raise ValueError(f"DataFrame must contain '{self.feature_cfg.sequences_column}' column")

        # Проверяем статические фичи
        static_features = self.feature_cfg.static_features
        missing_static = [col for col in static_features if col not in X.columns]
        if missing_static:
            raise ValueError(f"Отсутствуют необходимые статические фичи: {missing_static}")

        # Проверяем формат последовательностей
        first_sequence = X[self.feature_cfg.sequences_column].iloc[0]
        if not isinstance(first_sequence, (list, np.ndarray)):
            raise ValueError(f"'{self.feature_cfg.sequences_column}' column must contain sequences (lists or arrays)")

    def get_model_info(self) -> dict[str, str | list[str] | int | float | dict[str, int] | list[int]]:
        """
        Возвращает информацию о модели.

        Returns
        -------
        dict[str, str | list[str] | int | float | dict[str, int] | list[int]]
            Информация о модели
        """
        return {
            "model_path": str(self.onnx_model_path),
            "model_size_mb": self.onnx_model_path.stat().st_size / (1024 * 1024),
            "inputs": self._model_info["inputs"],
            "outputs": self._model_info["outputs"],
            "providers": self.ort_session.get_providers(),
            "feature_dims": self.feature_dims,
        }


class ONNXInferencePipelineFactory:
    """
    Фабрика для создания ONNX Inference Pipeline с предустановленными конфигурациями.
    """

    @staticmethod
    def create_fast_inference(onnx_model_path: str, sequence_config: dict | None = None) -> ONNXInferencePipeline:
        """
        Создает pipeline для быстрого инференса.

        Parameters
        ----------
        onnx_model_path : str
            Путь к ONNX модели
        sequence_config : dict, optional
            Конфигурация последовательностей

        Returns
        -------
        ONNXInferencePipeline
            Оптимизированный pipeline для быстрого инференса
        """
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 4
        session_options.inter_op_num_threads = 4

        return ONNXInferencePipeline(
            onnx_model_path=onnx_model_path,
            sequence_config=sequence_config,
            providers=["CPUExecutionProvider"],
            session_options=session_options,
        )

    @staticmethod
    def create_memory_efficient(onnx_model_path: str, sequence_config: dict | None = None) -> ONNXInferencePipeline:
        """
        Создает pipeline с оптимизацией памяти.

        Parameters
        ----------
        onnx_model_path : str
            Путь к ONNX модели
        sequence_config : dict, optional
            Конфигурация последовательностей

        Returns
        -------
        ONNXInferencePipeline
            Pipeline с оптимизацией памяти
        """
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1

        return ONNXInferencePipeline(
            onnx_model_path=onnx_model_path,
            sequence_config=sequence_config,
            providers=["CPUExecutionProvider"],
            session_options=session_options,
        )

    @staticmethod
    def create_gpu_inference(onnx_model_path: str, sequence_config: dict | None = None) -> ONNXInferencePipeline:
        """
        Создает pipeline для GPU инференса (если доступен).

        Parameters
        ----------
        onnx_model_path : str
            Путь к ONNX модели
        sequence_config : dict, optional
            Конфигурация последовательностей

        Returns
        -------
        ONNXInferencePipeline
            Pipeline для GPU инференса
        """
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        return ONNXInferencePipeline(
            onnx_model_path=onnx_model_path, sequence_config=sequence_config, providers=providers
        )
