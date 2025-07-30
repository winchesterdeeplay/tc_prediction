from typing import Literal

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from core.features import FeatureConfig

# Типы стратегий взвешивания
WeightingStrategy = Literal["balanced_horizon", "balanced_intensity", "uniform"]


def calculate_sample_weights(X: pd.DataFrame, strategy: WeightingStrategy = "balanced_horizon") -> np.ndarray:
    """
    Вычисляет веса для сэмплов на основе выбранной стратегии.

    Эта функция позволяет балансировать датасет по различным критериям
    для улучшения качества обучения моделей прогнозирования циклонов.

    Parameters
    ----------
    X : pd.DataFrame
        Датафрейм с данными, содержащий признаки циклонов
    strategy : WeightingStrategy, optional
        Стратегия взвешивания:
        - "balanced_horizon": балансировка по горизонту прогноза
        - "balanced_intensity": балансировка по интенсивности циклона
        - "uniform": равномерные веса для всех сэмплов
        по умолчанию "balanced_horizon"

    Returns
    -------
    np.ndarray
        Массив весов для каждого сэмпла в датасете

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'target_time_hours': [6, 12, 24, 48, 72],
    ...     'central_pressure_hpa': [950, 980, 1010, 1020, 1030]
    ... })
    >>> weights = calculate_sample_weights(df, strategy="balanced_horizon")
    >>> print(weights)
    [1.0 1.0 1.0 1.0 1.0]  # Все горизонты представлены равномерно

    Notes
    -----
    - При использовании "balanced_horizon" веса вычисляются на основе
      распределения горизонтов прогноза для обеспечения равномерного
      представления всех временных интервалов
    - При использовании "balanced_intensity" веса вычисляются на основе
      центрального давления циклона для балансировки по интенсивности
    - Если требуемые колонки отсутствуют в датасете, возвращаются
      единичные веса с предупреждением
    """
    feature_cfg = FeatureConfig()

    if strategy == "balanced_horizon":
        weights = _balance_by_horizon(X, feature_cfg)
    elif strategy == "balanced_intensity":
        weights = _balance_by_intensity(X)
    elif strategy == "uniform":
        weights = np.ones(len(X), dtype=np.float32)
    else:
        raise ValueError(f"Неизвестная стратегия взвешивания: {strategy}")

    return weights


def _balance_by_horizon(X: pd.DataFrame, feature_cfg: FeatureConfig) -> np.ndarray:
    """
    Балансирует датасет по горизонту прогноза.

    Parameters
    ----------
    X : pd.DataFrame
        Датафрейм с данными
    feature_cfg : FeatureConfig
        Конфигурация признаков

    Returns
    -------
    np.ndarray
        Веса для балансировки по горизонту
    """
    if feature_cfg.target_time_column in X.columns:
        horizons = X[feature_cfg.target_time_column].astype(int)
        weights = compute_sample_weight(class_weight="balanced", y=horizons)
        return weights.astype(np.float32)
    else:
        print(
            f"Предупреждение: колонка '{feature_cfg.target_time_column}' не найдена. " f"Используются единичные веса."
        )
        return np.ones(len(X), dtype=np.float32)


def _balance_by_intensity(X: pd.DataFrame) -> np.ndarray:
    """
    Балансирует датасет по интенсивности циклона.

    Parameters
    ----------
    X : pd.DataFrame
        Датафрейм с данными

    Returns
    -------
    np.ndarray
        Веса для балансировки по интенсивности
    """
    intensity_column = "central_pressure_hpa"
    if intensity_column in X.columns:
        # Дискретизируем давление для балансировки
        pressure_values = X[intensity_column].values
        # Создаем бины для давления (например, каждые 20 hPa)
        pressure_bins = pd.cut(pressure_values, bins=10, labels=False)
        weights = compute_sample_weight(class_weight="balanced", y=pressure_bins)
        return weights.astype(np.float32)
    else:
        print(f"Предупреждение: колонка '{intensity_column}' не найдена. " f"Используются единичные веса.")
        return np.ones(len(X), dtype=np.float32)
