from typing import Any

import numpy as np
import pandas as pd

from core.coordinates import CoordinateProcessor


def extract_current_coordinates(sequence: np.ndarray) -> tuple[float, float]:
    """
    Извлекает текущие координаты из последней непустой строки последовательности.

    Параметры:
    -----------
    sequence : np.ndarray
        Массив признаков с формой (time_steps, features)

    Возвращает:
    --------
    tuple[float, float]
        Текущие координаты (широта, долгота)
    """
    # Находим последнюю непустую строку
    non_zero_mask = (sequence != 0).any(axis=1)
    last_idx = int(np.where(non_zero_mask)[0][-1])
    return float(sequence[last_idx, 0]), float(sequence[last_idx, 1])


class ModelEvaluator:
    """
    Класс для оценки качества модели предсказания траекторий циклонов.

    Этот класс предоставляет методы для:
    1. Вычисления метрик качества предсказаний траекторий
    2. Анализа ошибок по различным горизонтам прогноза
    3. Обработки предсказаний координат из последовательностей
    """

    def __init__(self, coordinate_processor: CoordinateProcessor | None = None):
        """
        Инициализация оценщика модели.

        Параметры:
        -----------
        coordinate_processor : CoordinateProcessor, optional
            Экземпляр процессора координат. Если None, создается новый.
        """
        self.coord_processor = coordinate_processor or CoordinateProcessor()

    def evaluate_horizon(self, model: Any, X_horizon: pd.DataFrame, y_horizon: pd.DataFrame) -> dict[str, float]:
        """
        Оценивает производительность модели для одного горизонта прогноза.

        Параметры:
        -----------
        model : Any
            Обученная модель с методом predict
        X_horizon : pd.DataFrame
            Признаки для одного горизонта (должен содержать столбец 'sequences')
        y_horizon : pd.DataFrame
            Истинные значения для одного горизонта

        Возвращает:
        --------
        dict[str, float]
            Словарь с метриками оценки:
            - samples: Количество образцов
            - mean_km: Средняя ошибка в километрах
            - median_km: Медианная ошибка в километрах
            - max_km: Максимальная ошибка в километрах
            - p50: Процент предсказаний с ошибкой < 50км
            - p100: Процент предсказаний с ошибкой < 100км
            - p300: Процент предсказаний с ошибкой < 300км
        """
        if len(X_horizon) == 0:
            return {
                "samples": 0,
                "mean_km": np.nan,
                "median_km": np.nan,
                "max_km": np.nan,
                "p50": np.nan,
                "p100": np.nan,
                "p300": np.nan,
            }

        # Получаем предсказания
        predictions = model.predict(X_horizon)

        # Обрабатываем координаты
        lat_true, lon_true, lat_pred, lon_pred = self._process_predictions_from_sequences(
            X_horizon, y_horizon, predictions
        )

        # Вычисляем ошибки в километрах
        errors_km = self.coord_processor.haversine_distance(lat_true, lon_true, lat_pred, lon_pred)

        # Вычисляем метрики
        return {
            "samples": len(X_horizon),
            "mean_km": float(np.mean(errors_km)),
            "median_km": float(np.median(errors_km)),
            "max_km": float(np.max(errors_km)),
            "p50": float(np.mean(errors_km < 50) * 100),
            "p100": float(np.mean(errors_km < 100) * 100),
            "p300": float(np.mean(errors_km < 300) * 100),
        }

    def _process_predictions_from_sequences(
        self, X: pd.DataFrame, y: pd.DataFrame, predictions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Обрабатывает предсказания для последовательностей и преобразует в абсолютные координаты.

        Параметры:
        -----------
        X : pd.DataFrame
            Датафрейм признаков, содержащий столбец 'sequences'
        y : pd.DataFrame
            Целевой датафрейм со столбцами 'dlat_target' и 'dlon_target'
        predictions : np.ndarray
            Предсказания модели с формой (n_samples, 2) для дельт lat/lon

        Возвращает:
        --------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (lat_true, lon_true, lat_pred, lon_pred) - абсолютные координаты
        """
        lat_true = []
        lon_true = []
        lat_pred = []
        lon_pred = []

        for i in range(len(X)):
            # Извлекаем текущие координаты из последовательности
            curr_lat, curr_lon = extract_current_coordinates(X["sequences"].iloc[i])

            # Вычисляем предсказанные координаты
            pred_lat = curr_lat + predictions[i, 0]
            pred_lon = curr_lon + predictions[i, 1]

            # Истинные координаты
            true_lat = curr_lat + y.iloc[i]["dlat_target"]
            true_lon = curr_lon + y.iloc[i]["dlon_target"]

            lat_true.append(true_lat)
            lon_true.append(true_lon)
            lat_pred.append(pred_lat)
            lon_pred.append(pred_lon)

        return np.array(lat_true), np.array(lon_true), np.array(lat_pred), np.array(lon_pred)
