from abc import ABC, abstractmethod

import pandas as pd

from core.coordinates import CoordinateProcessor
from core.features import FeatureConfig

from .dataset_models import ProcessedDataset, SequenceConfig


class BaseDataProcessor(ABC):
    """
    Базовый абстрактный класс для процессоров данных.

    Содержит общие методы для предобработки данных и создания примеров.

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
        Инициализирует базовый процессор данных.

        Parameters
        ----------
        sequence_config : SequenceConfig | None
            Конфигурация последовательностей
        validate_data : bool
            Выполнять ли валидацию данных
        """
        self.feature_cfg = FeatureConfig()
        self.seq_config = sequence_config or SequenceConfig()
        self.validate_data = validate_data

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Предобрабатывает входные данные.

        Parameters
        ----------
        df : pd.DataFrame
            Входной датафрейм

        Returns
        -------
        pd.DataFrame
            Предобработанный датафрейм
        """
        if df.empty:
            raise ValueError("Входной DataFrame пуст")

        # Получаем только исходные фичи и метаданные (не производные и статические)
        required_cols = self.feature_cfg.raw_sequence_features + self.feature_cfg.metadata_features

        # Проверяем наличие колонок
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_cols}")

        # Фильтруем только нужные колонки
        df = df[required_cols].copy()

        # Конвертируем временные метки
        df["analysis_time"] = pd.to_datetime(df["analysis_time"])

        # Нормализуем долготы
        df["lon_deg"] = CoordinateProcessor.normalize_longitude(df["lon_deg"])

        # Удаляем строки с пропущенными значениями
        before_len = len(df)
        df = df.dropna(subset=required_cols)
        dropped = before_len - len(df)
        if dropped > 0:
            print(f"{self.__class__.__name__}: dropped {dropped} rows with missing values")

        # Валидируем данные
        if self.validate_data:
            # Валидируем только исходные фичи
            self.feature_cfg.validator.validate_raw(df)

            # Проверяем диапазоны координат
            self.feature_cfg.validator._check_column_ranges(df)

            print(f"Dataset validation: {len(df)} rows, {len(df.columns)} columns")

        # Сортируем по времени
        df = df.sort_values(["intl_id", "analysis_time"]).reset_index(drop=True)

        return df

    def _get_sequence_slice(self, cdf: pd.DataFrame, i: int) -> pd.DataFrame:
        """
        Получает срез последовательности.

        Parameters
        ----------
        cdf : pd.DataFrame
            Траектория циклона
        i : int
            Индекс текущей точки

        Returns
        -------
        pd.DataFrame
            Срез последовательности
        """
        seq_len = self.seq_config.max_history_length or len(cdf)
        seq_start = max(0, i + 1 - seq_len)
        return cdf.iloc[seq_start : i + 1]

    @abstractmethod
    def build_dataset(self, df: pd.DataFrame, horizon_hours: int) -> ProcessedDataset:
        """
        Строит датасет.

        Parameters
        ----------
        df : pd.DataFrame
            Входной датафрейм
        horizon_hours : int
            Горизонт прогноза в часах

        Returns
        -------
        ProcessedDataset
            Обработанный датасет
        """
        pass
