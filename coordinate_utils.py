import numpy as np
import pandas as pd


class CoordinateProcessor:
    @staticmethod
    def circ_diff(lon_from: pd.Series, lon_to: pd.Series) -> pd.Series:
        """
        Разница долгот с учётом 180°/‑180°.
        Возвращает диапазон (‑180°, +180°].

        Parameters
        ----------
        lon_from : pd.Series
            Исходная долгота
        lon_to : pd.Series
            Целевая долгота

        Returns
        -------
        pd.Series
            Разница долгот в диапазоне (-180°, +180°]
        """
        return ((lon_to - lon_from + 180) % 360) - 180

    @staticmethod
    def normalize_longitude(lon: pd.Series) -> pd.Series:
        """
        Нормализует долготу в диапазон (-180°, +180°].
        Преобразует долготы из диапазона [0°, 360°] в [-180°, +180°].

        Parameters
        ----------
        lon : pd.Series
            Долгота в любом диапазоне

        Returns
        -------
        pd.Series
            Нормализованная долгота в диапазоне (-180°, +180°]
        """
        return ((lon + 180) % 360) - 180

    @staticmethod
    def circ_add(lon_base: np.ndarray, dlon: np.ndarray) -> np.ndarray:
        """
        Сложение долготы с учётом перехода через 180°/‑180°.

        Parameters
        ----------
        lon_base : np.ndarray
            Базовая долгота
        dlon : np.ndarray
            Приращение долготы

        Returns
        -------
        np.ndarray
            Результирующая долгота в диапазоне (-180°, +180°]
        """
        return ((lon_base + dlon + 180) % 360) - 180

    def apply_coordinate_deltas(
        self, lat_current: np.ndarray, lon_current: np.ndarray, dlat_pred: np.ndarray, dlon_pred: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Применяет предсказанные смещения к текущим координатам.

        Parameters
        ----------
        lat_current : np.ndarray
            Текущая широта
        lon_current : np.ndarray
            Текущая долгота
        dlat_pred : np.ndarray
            Предсказанное смещение по широте
        dlon_pred : np.ndarray
            Предсказанное смещение по долготе

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Предсказанные координаты (lat_pred, lon_pred)
        """
        lat_pred = lat_current + dlat_pred
        lon_pred = self.circ_add(lon_current, dlon_pred)
        return lat_pred, lon_pred

    def process_predictions(
        self, X_test: pd.DataFrame, y_true: pd.DataFrame, predictions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Обрабатывает предсказания модели и возвращает координаты.

        Parameters
        ----------
        X_test : pd.DataFrame
            Тестовые признаки (содержит текущие координаты)
        y_true : pd.DataFrame
            Истинные значения (содержит dlat_target, dlon_target)
        predictions : np.ndarray
            Предсказания модели (dlat_pred, dlon_pred)

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            lat_true, lon_true, lat_pred, lon_pred
        """
        if predictions.shape[1] != 2:
            raise ValueError(f"Ожидается 2 колонки в предсказаниях, получено: {predictions.shape[1]}")

        dlat_pred, dlon_pred = predictions.T

        lat_current = X_test["latitude_prev"].values
        lon_current = X_test["longitude_prev"].values

        lat_true = lat_current + y_true["dlat_target"].values
        lon_true = self.circ_add(lon_current, y_true["dlon_target"].values)

        lat_pred, lon_pred = self.apply_coordinate_deltas(lat_current, lon_current, dlat_pred, dlon_pred)

        return lat_true, lon_true, lat_pred, lon_pred
