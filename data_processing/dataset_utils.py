import pandas as pd


def split_data_by_years(
    X: pd.DataFrame, y: pd.DataFrame, times: pd.Series, train_max_year: int, val_max_year: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Разбивает датасет по годам на обучающую, валидационную и тестовую выборки.

    Эта функция помогает правильно разделить данные по времени, чтобы
    модель обучалась на прошлых данных, а тестировалась на будущих.
    Это важно для реалистичной оценки качества прогнозов.

    Parameters
    ----------
    X : pd.DataFrame
        Датафрейм с фичами - входные данные для модели
    y : pd.DataFrame
        Датафрейм с целевыми переменными - что предсказываем
    times : pd.Series
        Временные метки для каждого наблюдения (должен быть datetime)
    train_max_year : int
        Максимальный год для обучающей выборки. Все данные до этого года
        (включительно) идут в train.
    val_max_year : int
        Максимальный год для валидационной выборки. Данные между train_max_year
        и val_max_year идут в validation, остальные в test.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Кортеж из 6 датафреймов: (X_train, y_train, X_val, y_val, X_test, y_test)

    Examples
    --------
    >>> import pandas as pd
    >>> from datetime import datetime
    >>>
    >>> # Создаем тестовые данные
    >>> dates = pd.date_range('2015-01-01', '2020-12-31', freq='D')
    >>> X = pd.DataFrame({'feature1': range(len(dates))}, index=dates)
    >>> y = pd.DataFrame({'target': range(len(dates))}, index=dates)
    >>>
    >>> # Разбиваем по годам
    >>> X_train, y_train, X_val, y_val, X_test, y_test = split_data_by_years(
    ...     X, y, pd.Series(dates), train_max_year=2017, val_max_year=2019
    ... )
    >>>
    >>> print(f"Train: {len(X_train)} samples (2015-2017)")
    >>> print(f"Val: {len(X_val)} samples (2018-2019)")
    >>> print(f"Test: {len(X_test)} samples (2020)")

    Notes
    -----
    - Временные метки должны быть в формате datetime
    - Годы должны идти в хронологическом порядке: train_max_year <= val_max_year
    - Если val_max_year больше максимального года в данных, test будет пустым
    """
    # Убеждаемся, что индексы совпадают
    if not X.index.equals(y.index):
        raise ValueError("Индексы X и y должны совпадать")

    if not X.index.equals(times.index):
        raise ValueError("Индексы X и times должны совпадать")

    # Создаем маски для каждого разбиения
    train_mask = times.dt.year <= train_max_year
    val_mask = (times.dt.year > train_max_year) & (times.dt.year <= val_max_year)
    test_mask = times.dt.year > val_max_year

    # Разбиваем данные по маскам
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    return X_train, y_train, X_val, y_val, X_test, y_test
