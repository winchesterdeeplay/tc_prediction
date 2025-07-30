import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from core.features import FeatureConfig
from models.dataset import CycloneDataset, collate_variable_length


class CycloneDataModule(pl.LightningDataModule):
    """
    Модуль данных для работы с циклонными данными в PyTorch Lightning.

    Этот класс управляет загрузкой, предобработкой и аугментацией данных
    для тренировки моделей прогнозирования циклонов. Поддерживает
    нормализацию последовательностей, аугментацию данных и различные
    стратегии перемешивания.

    Attributes
    ----------
    X_train : pd.DataFrame
        Тренировочные признаки
    y_train : pd.Series
        Тренировочные метки
    X_val : pd.DataFrame
        Валидационные признаки
    y_val : pd.Series
        Валидационные метки
    sample_weight : list[float] | None
        Веса сэмплов для взвешенного обучения
    batch_size : int
        Размер батча для обучения
    num_workers : int
        Количество воркеров для загрузки данных
    shuffle_sequences : bool
        Перемешивать ли последовательности внутри батча
    shuffle_batch : bool
        Перемешивать ли элементы внутри батча
    shuffle_dataset : bool
        Перемешивать ли датасет при создании
    normalize_sequences : bool
        Нормализовать ли последовательности
    augment_data : bool
        Применять ли аугментацию данных
    noise_std : float
        Стандартное отклонение шума для аугментации
    dropout_rate : float
        Коэффициент dropout для аугментации
    sequence_mean : float | None
        Среднее значение для нормализации последовательностей
    sequence_std : float | None
        Стандартное отклонение для нормализации последовательностей
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        *,
        sample_weight: list[float] | None = None,
        batch_size: int = 64,
        num_workers: int = 0,
        shuffle_sequences: bool = False,
        shuffle_batch: bool = False,
        shuffle_dataset: bool = False,
        normalize_sequences: bool = False,
        augment_data: bool = False,
        noise_std: float = 0.01,
        dropout_rate: float = 0.1,
    ) -> None:
        """
        Инициализация модуля данных.

        Parameters
        ----------
        X_train : pd.DataFrame
            Тренировочные признаки
        y_train : pd.Series
            Тренировочные метки
        X_val : pd.DataFrame
            Валидационные признаки
        y_val : pd.Series
            Валидационные метки
        sample_weight : list[float] | None, optional
            Веса сэмплов для взвешенного обучения, по умолчанию None
        batch_size : int, optional
            Размер батча для обучения, по умолчанию 64
        num_workers : int, optional
            Количество воркеров для загрузки данных, по умолчанию 0
        shuffle_sequences : bool, optional
            Перемешивать ли последовательности внутри батча, по умолчанию False
        shuffle_batch : bool, optional
            Перемешивать ли элементы внутри батча, по умолчанию False
        shuffle_dataset : bool, optional
            Перемешивать ли датасет при создании, по умолчанию False
        normalize_sequences : bool, optional
            Нормализовать ли последовательности, по умолчанию True
        augment_data : bool, optional
            Применять ли аугментацию данных, по умолчанию True
        noise_std : float, optional
            Стандартное отклонение шума для аугментации, по умолчанию 0.01
        dropout_rate : float, optional
            Коэффициент dropout для аугментации, по умолчанию 0.1
        """
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.sample_weight = sample_weight
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_sequences = shuffle_sequences
        self.shuffle_batch = shuffle_batch
        self.shuffle_dataset = shuffle_dataset
        self.normalize_sequences = normalize_sequences
        self.augment_data = augment_data
        self.noise_std = noise_std
        self.dropout_rate = dropout_rate

        # Статистики для нормализации
        self.sequence_mean: float | None = None
        self.sequence_std: float | None = None

    def setup(self, stage: str | None = None) -> None:
        """
        Настройка датасетов для тренировки и валидации.

        Этот метод вызывается PyTorch Lightning для подготовки данных.
        Создает тренировочный и валидационный датасеты с применением
        нормализации и других преобразований.

        Parameters
        ----------
        stage : str | None, optional
            Стадия обучения ('fit', 'validate', 'test', 'predict'), по умолчанию None
        """
        # Создаем статические фичи, включая горизонты прогноза
        feature_cfg = FeatureConfig()
        static_cols = feature_cfg.static_features

        static_features_train = self.X_train[static_cols].values
        static_features_val = self.X_val[static_cols].values

        # Извлекаем горизонты отдельно для loss функций
        horizon_hours_train = self.X_train[feature_cfg.target_time_column].values
        horizon_hours_val = self.X_val[feature_cfg.target_time_column].values

        # Нормализуем последовательности если включено
        if self.normalize_sequences:
            self._compute_normalization_stats()
            sequences_train = self._normalize_sequences(self.X_train[feature_cfg.sequences_column].tolist())
            sequences_val = self._normalize_sequences(self.X_val[feature_cfg.sequences_column].tolist())
        else:
            sequences_train = self.X_train[feature_cfg.sequences_column].tolist()
            sequences_val = self.X_val[feature_cfg.sequences_column].tolist()

        self.train_dataset = CycloneDataset(
            sequences_train,
            static_features_train,
            self.y_train.values,
            horizon_hours_train,
            np.array(self.sample_weight) if self.sample_weight is not None else None,
            shuffle_dataset=self.shuffle_dataset,
        )
        self.val_dataset = CycloneDataset(
            sequences_val,
            static_features_val,
            self.y_val.values,
            horizon_hours_val,
            None,  # sample_weight для валидации
            shuffle_dataset=False,  # Не перемешиваем валидационный датасет
        )

    def _compute_normalization_stats(self) -> None:
        """
        Вычисляет статистики для нормализации последовательностей.

        Вычисляет среднее значение и стандартное отклонение по всем
        последовательностям в тренировочном наборе данных для последующей
        нормализации.
        """
        feature_cfg = FeatureConfig()
        all_sequences = self.X_train[feature_cfg.sequences_column].tolist()

        # Собираем все значения
        all_values: list[float] = []
        for seq in all_sequences:
            all_values.extend(seq.flatten())

        all_values_array = np.array(all_values)
        self.sequence_mean = float(np.mean(all_values_array))
        self.sequence_std = float(np.std(all_values_array))

        # Избегаем деления на ноль
        if self.sequence_std < 1e-8:
            self.sequence_std = 1.0

    def _normalize_sequences(self, sequences: list[np.ndarray]) -> list[np.ndarray]:
        """
        Нормализует последовательности используя предвычисленные статистики.

        Parameters
        ----------
        sequences : List[np.ndarray]
            Список последовательностей для нормализации

        Returns
        -------
        List[np.ndarray]
            Список нормализованных последовательностей
        """
        if self.sequence_mean is None or self.sequence_std is None:
            return sequences

        normalized_sequences: list[np.ndarray] = []
        for seq in sequences:
            normalized_seq = (seq - self.sequence_mean) / self.sequence_std
            normalized_sequences.append(normalized_seq)

        return normalized_sequences

    def _augment_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Применяет аугментацию к последовательности.

        Добавляет шум и применяет dropout к временным шагам для
        улучшения обобщающей способности модели.

        Parameters
        ----------
        sequence : np.ndarray
            Исходная последовательность

        Returns
        -------
        np.ndarray
            Аугментированная последовательность
        """
        if not self.augment_data:
            return sequence

        # Добавляем небольшой шум
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, sequence.shape)
            sequence = sequence + noise

        # Применяем dropout к некоторым временным шагам
        if self.dropout_rate > 0:
            mask = np.random.random(sequence.shape[0]) > self.dropout_rate
            if np.sum(mask) > 1:  # Оставляем минимум 1 шаг
                sequence = sequence[mask]

        return sequence

    def train_dataloader(self) -> DataLoader:
        """
        Создает DataLoader для тренировочных данных.

        Returns
        -------
        DataLoader
            DataLoader для тренировочных данных с перемешиванием
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_variable_length(
                batch, shuffle_sequences=self.shuffle_sequences, shuffle_batch=self.shuffle_batch
            ),
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Создает DataLoader для валидационных данных.

        Returns
        -------
        DataLoader
            DataLoader для валидационных данных без перемешивания
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_variable_length(
                batch, shuffle_sequences=False, shuffle_batch=False  # Не перемешиваем в валидации
            ),
            num_workers=self.num_workers,
        )
