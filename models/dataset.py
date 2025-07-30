from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class CycloneDataset(Dataset):
    """
    Датасет для последовательностей циклонов переменной длины с разделенными фичами.

    Параметры:
    ----------
    sequences : list[np.ndarray]
        Список последовательностей переменной длины (только последовательностные фичи)
    static_features : np.ndarray
        Статические фичи для каждого семпла
    y : np.ndarray
        Целевые значения (изменения координат)
    sample_weight : np.ndarray, optional
        Веса семплов для балансировки
    """

    def __init__(
        self,
        sequences: list[np.ndarray],
        static_features: np.ndarray,
        y: np.ndarray,
        horizon_hours: np.ndarray,
        sample_weight: np.ndarray | None = None,
        shuffle_dataset: bool = False,
    ):
        """
        Инициализация датасета.

        Параметры:
        ----------
        sequences : list[np.ndarray]
            Список последовательностей переменной длины
        static_features : np.ndarray
            Статические фичи [n_samples, n_static_features]
        y : np.ndarray
            Целевые значения (dlat, dlon)
        horizon_hours : np.ndarray
            Горизонты прогноза в часах [n_samples]
        sample_weight : np.ndarray, optional
            Веса семплов
        shuffle_dataset : bool, optional
            Перемешивать ли весь датасет при инициализации
        """
        # Сохраняем как список массивов разной длины
        self.sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
        self.static_features = torch.tensor(static_features, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.horizon_hours = torch.tensor(horizon_hours, dtype=torch.float32)

        if sample_weight is None:
            sample_weight = np.ones(len(sequences), dtype=np.float32)
        self.sample_weight = torch.tensor(sample_weight, dtype=torch.float32)

        # Перемешиваем весь датасет, если нужно
        if shuffle_dataset:
            self._shuffle_dataset()

    def _shuffle_dataset(self) -> None:
        """Перемешивает весь датасет."""
        indices = torch.randperm(len(self.sequences))
        self.sequences = [self.sequences[i] for i in indices]
        self.static_features = self.static_features[indices]
        self.y = self.y[indices]
        self.horizon_hours = self.horizon_hours[indices]
        self.sample_weight = self.sample_weight[indices]

    def __len__(self) -> int:
        """Возвращает количество семплов в датасете."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple:
        """
        Возвращает один семпл из датасета.

        Параметры:
        ----------
        idx : int
            Индекс семпла

        Возвращает:
        ----------
        tuple
            (sequence, static_features, target, weight, horizon_hours)
        """
        return (
            self.sequences[idx],
            self.static_features[idx],
            self.y[idx],
            self.sample_weight[idx],
            self.horizon_hours[idx],
        )


def collate_variable_length(batch: Any, shuffle_sequences: bool = False, shuffle_batch: bool = False) -> tuple:
    """
    Collate функция для батчей с переменной длиной последовательностей.
    Выполняет динамический паддинг до максимальной длины в батче.

    Параметры:
    ----------
    batch : list
        Список кортежей (sequence, static_features, target, weight, horizon_hours) из Dataset
    shuffle_sequences : bool
        Перемешивать ли последовательности внутри каждого образца
    shuffle_batch : bool
        Перемешивать ли образцы в батче

    Возвращает:
    ----------
    tuple
        (padded_sequences, static_features, targets, weights, horizon_hours, sequence_lengths)
    """
    # Извлекаем данные из батча
    sequences_tuple, static_features_tuple, targets_tuple, weights_tuple, horizon_hours_tuple = zip(*batch)

    # Конвертируем в списки
    sequences = list(sequences_tuple)
    static_features = list(static_features_tuple)
    targets = list(targets_tuple)
    weights = list(weights_tuple)
    horizon_hours = list(horizon_hours_tuple)

    # Перемешиваем образцы в батче, если нужно
    if shuffle_batch:
        indices = torch.randperm(len(batch))
        sequences = [sequences[i] for i in indices]
        static_features = [static_features[i] for i in indices]
        targets = [targets[i] for i in indices]
        weights = [weights[i] for i in indices]
        horizon_hours = [horizon_hours[i] for i in indices]

    # Перемешиваем последовательности внутри каждого образца, если нужно
    if shuffle_sequences:
        shuffled_sequences = []
        for seq in sequences:
            if seq.size(0) > 1:  # Перемешиваем только если длина > 1
                # Создаем случайную перестановку индексов
                perm = torch.randperm(seq.size(0))
                shuffled_seq = seq[perm]
                shuffled_sequences.append(shuffled_seq)
            else:
                shuffled_sequences.append(seq)
        sequences = shuffled_sequences

    # Определяем максимальную длину последовательности
    max_len = max(seq.size(0) for seq in sequences)

    # Создаем пустые тензоры для паддинга
    padded_sequences = torch.zeros(len(batch), max_len, sequences[0].size(1))
    sequence_lengths = torch.zeros(len(batch), dtype=torch.long)

    # Заполняем паддингами
    for i, seq in enumerate(sequences):
        seq_len = seq.size(0)
        padded_sequences[i, :seq_len] = seq
        sequence_lengths[i] = seq_len

    # Конвертируем остальные данные в тензоры
    static_features_tensor = torch.stack(static_features)
    targets_tensor = torch.stack(targets)
    weights_tensor = torch.stack(weights)
    horizon_hours_tensor = torch.stack(horizon_hours)

    return padded_sequences, static_features_tensor, targets_tensor, weights_tensor, horizon_hours_tensor, sequence_lengths
