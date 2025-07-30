import pandas as pd

from .base_processor import BaseDataProcessor
from .dataset_models import ProcessedDataset, SequenceConfig


class InferenceDataProcessor(BaseDataProcessor):
    """
    Процессор данных для инференса в продакшене.

    Упрощенная версия DataProcessor без параметров обучения.
    Оптимизирован для быстрого инференса в продакшене.

    Attributes
    ----------
    feature_cfg : FeatureConfig
        Конфигурация фич с реестром
    seq_config : SequenceConfig
        Конфигурация последовательностей
    validate_data : bool
        Флаг валидации данных
    """

    def __init__(
        self,
        sequence_config: SequenceConfig | None = None,
        validate_data: bool = True,
    ):
        """
        Инициализирует процессор данных для инференса.

        Parameters
        ----------
        sequence_config : SequenceConfig | None
            Конфигурация последовательностей
        validate_data : bool
            Выполнять ли валидацию данных
        """
        super().__init__(sequence_config, validate_data)

    def build_dataset(self, df: pd.DataFrame) -> ProcessedDataset:
        """
        Строит датасет для инференса.

        Parameters
        ----------
        df : pd.DataFrame
            Входной датафрейм

        Returns
        -------
        ProcessedDataset
            Датасет для инференса
        """
        return self._build_dataset(df, horizon_hours=24)

    def _build_dataset(self, df: pd.DataFrame, horizon_hours: int = 24) -> ProcessedDataset:
        """
        Строит датасет для инференса.

        Parameters
        ----------
        df : pd.DataFrame
            Входной датафрейм
        horizon_hours : int
            Горизонт прогноза в часах

        Returns
        -------
        ProcessedDataset
            Датасет для инференса
        """
        # Предобрабатываем данные
        df = self.preprocess_data(df)

        # Создаем примеры для инференса
        examples = self._create_examples(df, horizon_hours)

        if not examples:
            raise ValueError("Не найдено валидных примеров для инференса")

        # Создаем финальный датафрейм
        df_final = pd.DataFrame(examples)

        # Валидируем готовый датасет
        if self.validate_data:
            self.feature_cfg.validator.validate_sequences_format(df_final)

        return ProcessedDataset(
            X=df_final,
            y=None,
            times=df_final["analysis_time"],
            intl_ids=df_final["intl_id"],
            storm_names=df_final["storm_name"],
        )

    def _create_examples(self, df: pd.DataFrame, horizon_hours: int) -> list[dict]:
        """
        Создает примеры для инференса.

        Parameters
        ----------
        df : pd.DataFrame
            Предобработанный датафрейм
        horizon_hours : int
            Горизонт прогноза

        Returns
        -------
        list[dict]
            Список примеров для инференса
        """
        examples = []

        for cid, cdf in df.groupby("intl_id"):
            cdf = cdf.reset_index(drop=True)

            if len(cdf) < self.seq_config.min_history_length:
                continue

            # Берем только последнюю точку
            i = len(cdf) - 1
            curr = cdf.iloc[i]

            example = self._create_single_example(cdf, i, curr, horizon_hours)
            if example:
                examples.append(example)

        return examples

    def _create_single_example(self, cdf: pd.DataFrame, i: int, curr: pd.Series, horizon: int) -> dict | None:
        """
        Создает один пример для инференса.

        Parameters
        ----------
        cdf : pd.DataFrame
            Траектория циклона
        i : int
            Индекс текущей точки
        curr : pd.Series
            Текущая точка
        horizon : int
            Горизонт прогноза

        Returns
        -------
        dict | None
            Пример или None
        """
        # Получаем последовательность
        seq_df = self._get_sequence_slice(cdf, i)

        # Создаем последовательностные фичи
        sequences = self.feature_cfg.create_enhanced_sequence(seq_df)

        # Вычисляем статические фичи
        static_features = self.feature_cfg.compute_static_features(curr, horizon)

        # Создаем пример с исходными координатами
        example = {
            "sequences": sequences,
            "analysis_time": curr["analysis_time"],
            "intl_id": curr["intl_id"],
            "storm_name": curr.get("storm_name", "unknown"),
            "lat_deg": curr["lat_deg"],  # Исходная широта
            "lon_deg": curr["lon_deg"],  # Исходная долгота
            **static_features,
        }

        return example
