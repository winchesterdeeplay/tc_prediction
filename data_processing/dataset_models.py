from dataclasses import dataclass

import pandas as pd


@dataclass
class ProcessedDataset:
    """
    Контейнер для обработанного датасета.

    Содержит все необходимые компоненты для обучения и валидации модели.
    Этот класс обеспечивает структурированный доступ к данным после обработки
    и может использоваться как для обучения, так и для инференса.

    Attributes
    ----------
    X : pd.DataFrame
        Датафрейм с входными фичами (последовательности и статические фичи).
        Содержит колонки: sequences, target_time_hours, day_of_year_sin/cos,
        month_of_year_sin/cos, analysis_time, intl_id, storm_name
    y : pd.DataFrame | None
        Датафрейм с целевыми переменными (dlat_target, dlon_target) или None для инференса.
        Для обучения содержит дельты координат для заданного горизонта прогноза.
    times : pd.Series
        Временные метки для каждого примера (analysis_time)
    intl_ids : pd.Series
        Идентификаторы циклонов для каждого примера
    storm_names : pd.Series
        Названия циклонов для каждого примера

    Examples
    --------
    >>> dataset = processor.build_dataset(cyclone_data)
    >>> print(f"Input features: {dataset.X.shape}")
    >>> print(f"Target variables: {dataset.y.shape if dataset.y is not None else 'None'}")
    >>> print(f"Number of samples: {len(dataset.times)}")
    """

    X: pd.DataFrame
    y: pd.DataFrame | None
    times: pd.Series
    intl_ids: pd.Series
    storm_names: pd.Series


@dataclass
class SequenceConfig:
    """
    Конфигурация для генерации последовательностей.

    Определяет параметры построения последовательностей из траекторий циклонов.
    Позволяет контролировать длину истории, используемой для прогнозирования.

    Parameters
    ----------
    min_history_length : int, default=2
        Минимальная длина последовательности. Последовательности короче этого
        значения будут пропущены. Должно быть >= 2 для корректного вычисления
        производных фич (скорость, направление).
    max_history_length : int | None, default=None
        Максимальная длина последовательности. Если None, используется вся
        доступная история циклона. Ограничивает количество исторических точек
        для ускорения обработки и контроля размера модели.

    Examples
    --------
    >>> # Использование всей доступной истории
    >>> config = SequenceConfig(min_history_length=3)
    >>>
    >>> # Ограничение истории последними 10 точками
    >>> config = SequenceConfig(min_history_length=3, max_history_length=10)
    >>>
    >>> # Минимальная конфигурация для быстрой обработки
    >>> config = SequenceConfig(min_history_length=2, max_history_length=5)
    """

    min_history_length: int = 1
    max_history_length: int | None = None

    def __post_init__(self) -> None:
        if self.min_history_length < 2:
            raise ValueError("min_history_length должен быть >= 2 для корректного " "вычисления производных фич")

        if self.max_history_length is not None and self.max_history_length < self.min_history_length:
            raise ValueError("max_history_length должен быть >= min_history_length")
