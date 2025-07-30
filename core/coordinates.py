from typing import Any

import numpy as np
import pandas as pd


class CoordinateProcessor:
    """
    Помогает работать с географическими координатами - вычислять расстояния,
    нормализовать долготы и находить направления между точками.
    """

    EARTH_RADIUS_KM = 6371.0  # Радиус Земли в километрах

    @staticmethod
    def haversine_distance(
        lat1: np.ndarray | float, lon1: np.ndarray | float, lat2: np.ndarray | float, lon2: np.ndarray | float
    ) -> np.ndarray | float:
        """
        Считает расстояние между двумя точками на Земле по прямой линии.

        Использует формулу гаверсинуса - она даёт точные результаты
        для любых расстояний и не ломается на маленьких дистанциях.

        Параметры:
        ----------
        lat1 : np.ndarray | float
            Широта первой точки (-90 до 90 градусов)
        lon1 : np.ndarray | float
            Долгота первой точки (-180 до 180 градусов)
        lat2 : np.ndarray | float
            Широта второй точки (-90 до 90 градусов)
        lon2 : np.ndarray | float
            Долгота второй точки (-180 до 180 градусов)

        Возвращает:
        ----------
        np.ndarray | float
            Расстояние в километрах. Если передали числа - вернёт число,
            если массивы - вернёт массив.

        Примеры:
        --------
        >>> CoordinateProcessor.haversine_distance(55.7558, 37.6176, 59.9311, 30.3609)
        634.5  # От Москвы до Питера

        >>> lats = np.array([55.7558, 59.9311])
        >>> lons = np.array([37.6176, 30.3609])
        >>> CoordinateProcessor.haversine_distance(lats[0], lons[0], lats[1], lons[1])
        634.5
        """
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        result: np.ndarray | float = CoordinateProcessor.EARTH_RADIUS_KM * c
        return result

    @staticmethod
    def compute_velocity(
        lats_prev: np.ndarray,
        lons_prev: np.ndarray,
        lats_curr: np.ndarray,
        lons_curr: np.ndarray,
        time_diffs_hours: np.ndarray,
    ) -> np.ndarray | float:
        """
        Вычисляет скорость движения циклона в км/ч между двумя точками.

        Использует формулу гаверсинуса для вычисления расстояния по сферическим координатам
        и делит на временной интервал для получения скорости.

        Parameters
        ----------
        lats_prev : np.ndarray
            Широты предыдущих точек в градусах
        lons_prev : np.ndarray
            Долготы предыдущих точек в градусах
        lats_curr : np.ndarray
            Широты текущих точек в градусах
        lons_curr : np.ndarray
            Долготы текущих точек в градусах
        time_diffs_hours : np.ndarray
            Временные интервалы в часах между точками

        Returns
        -------
        np.ndarray
            Скорости движения в км/ч

        Notes
        -----
        - Использует радиус Земли 6371 км
        - Обрабатывает случаи с нулевыми временными интервалами (заменяет на 1 час)
        - Все входные массивы должны иметь одинаковую длину
        """
        # Конвертируем в радианы
        lat1_rad = np.radians(lats_prev)
        lat2_rad = np.radians(lats_curr)
        dlat = np.radians(lats_curr - lats_prev)
        dlon = np.radians(lons_curr - lons_prev)

        # Формула гаверсинуса
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

        # Расстояние в километрах
        distances_km = CoordinateProcessor.EARTH_RADIUS_KM * c

        # Избегаем деления на ноль
        safe_time_diffs = np.where(time_diffs_hours == 0, 1.0, time_diffs_hours)

        result: np.ndarray | float = distances_km / safe_time_diffs
        return result

    @staticmethod
    def compute_bearing(
        lats_prev: np.ndarray, lons_prev: np.ndarray, lats_curr: np.ndarray, lons_curr: np.ndarray
    ) -> np.ndarray | float:
        """
        Вычисляет направление движения циклона в градусах между двумя точками.

        Использует формулу для вычисления азимута между двумя точками на сфере.
        Возвращает угол в градусах от 0 до 360, где 0° - север, 90° - восток.

        Parameters
        ----------
        lats_prev : np.ndarray
            Широты предыдущих точек в градусах
        lons_prev : np.ndarray
            Долготы предыдущих точек в градусах
        lats_curr : np.ndarray
            Широты текущих точек в градусах
        lons_curr : np.ndarray
            Долготы текущих точек в градусах

        Returns
        -------
        np.ndarray
            Направления движения в градусах (0-360)

        Notes
        -----
        - 0° соответствует движению на север
        - 90° соответствует движению на восток
        - 180° соответствует движению на юг
        - 270° соответствует движению на запад
        - Все входные массивы должны иметь одинаковую длину
        """
        # Конвертируем в радианы
        lat1_rad = np.radians(lats_prev)
        lat2_rad = np.radians(lats_curr)
        dlon = np.radians(lons_curr - lons_prev)

        # Вычисляем азимут
        y = np.sin(dlon) * np.cos(lat2_rad)
        x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
        bearings_rad = np.arctan2(y, x)

        # Конвертируем в градусы и нормализуем к диапазону [0, 360)
        bearings_deg = np.degrees(bearings_rad)
        result: np.ndarray | float = (bearings_deg + 360) % 360
        return result

    @staticmethod
    def compute_pressure_change(
        pressures_prev: np.ndarray, pressures_curr: np.ndarray, time_diffs_hours: np.ndarray
    ) -> np.ndarray | float:
        """
        Вычисляет изменение давления в гПа/ч между двумя точками.

        Вычисляет скорость изменения центрального давления циклона.
        Положительные значения означают рост давления, отрицательные - падение.

        Parameters
        ----------
        pressures_prev : np.ndarray
            Давления в предыдущих точках в гПа
        pressures_curr : np.ndarray
            Давления в текущих точках в гПа
        time_diffs_hours : np.ndarray
            Временные интервалы в часах между точками

        Returns
        -------
        np.ndarray
            Изменения давления в гПа/ч

        Notes
        -----
        - Положительные значения: давление растет (циклон ослабевает)
        - Отрицательные значения: давление падает (циклон усиливается)
        - Избегает деления на ноль, заменяя нулевые интервалы на 1 час
        - Все входные массивы должны иметь одинаковую длину
        """
        # Избегаем деления на ноль
        safe_time_diffs = np.where(time_diffs_hours == 0, 1.0, time_diffs_hours)

        result: np.ndarray | float = (pressures_curr - pressures_prev) / safe_time_diffs
        return result

    @staticmethod
    def compute_acceleration(velocities_kmh: np.ndarray, time_diffs_accel: np.ndarray) -> np.ndarray | float:
        """
        Вычисляет ускорение движения циклона в км/ч².

        Вычисляет скорость изменения скорости движения между последовательными точками.

        Parameters
        ----------
        velocities_kmh : np.ndarray
            Скорости движения в км/ч
        time_diffs_accel : np.ndarray
            Временные интервалы в часах между точками для вычисления ускорения

        Returns
        -------
        np.ndarray
            Ускорения в км/ч²

        Notes
        -----
        - Положительные значения: скорость растет (ускорение)
        - Отрицательные значения: скорость падает (замедление)
        - Избегает деления на ноль, заменяя нулевые интервалы на 1 час
        - Длина результата на 1 меньше длины входного массива скоростей
        """
        # Избегаем деления на ноль
        safe_time_diffs = np.where(time_diffs_accel == 0, 1.0, time_diffs_accel)

        result: np.ndarray | float = (velocities_kmh[1:] - velocities_kmh[:-1]) / safe_time_diffs
        return result

    @staticmethod
    def compute_angular_velocity(bearings_deg: np.ndarray, time_diffs_accel: np.ndarray) -> np.ndarray | float:
        """
        Вычисляет угловую скорость поворота циклона в градусах/ч.

        Вычисляет скорость изменения направления движения между последовательными точками.
        Корректно обрабатывает переходы через 0°/360°.

        Parameters
        ----------
        bearings_deg : np.ndarray
            Направления движения в градусах (0-360)
        time_diffs_accel : np.ndarray
            Временные интервалы в часах между точками для вычисления угловой скорости

        Returns
        -------
        np.ndarray
            Угловые скорости в градусах/ч

        Notes
        -----
        - Положительные значения: поворот по часовой стрелке
        - Отрицательные значения: поворот против часовой стрелки
        - Корректно обрабатывает переходы через 0°/360°
        - Избегает деления на ноль, заменяя нулевые интервалы на 1 час
        - Длина результата на 1 меньше длины входного массива направлений
        """
        # Вычисляем разности направлений
        bearing_diffs = bearings_deg[1:] - bearings_deg[:-1]

        # Корректируем переходы через 0°/360°
        bearing_diffs = np.where(bearing_diffs > 180, bearing_diffs - 360, bearing_diffs)
        bearing_diffs = np.where(bearing_diffs < -180, bearing_diffs + 360, bearing_diffs)

        # Избегаем деления на ноль
        safe_time_diffs = np.where(time_diffs_accel == 0, 1.0, time_diffs_accel)

        result: np.ndarray | float = bearing_diffs / safe_time_diffs
        return result

    @staticmethod
    def normalize_longitude(lon: pd.Series) -> pd.Series:
        """
        Приводит долготы к нормальному виду - от -180 до 180 градусов.

        Если долгота выходит за эти пределы, переводит её в эквивалентное
        значение. Например, 200° станет -160°, а -200° станет 160°.

        Параметры:
        ----------
        lon : pd.Series
            Долготы в градусах

        Возвращает:
        ----------
        pd.Series
            Долготы в диапазоне от -180 до 180 градусов

        Примеры:
        --------
        >>> import pandas as pd
        >>> lons = pd.Series([200, -200, 180, -180, 0])
        >>> CoordinateProcessor.normalize_longitude(lons)
        0   -160
        1    160
        2    180
        3   -180
        4      0
        dtype: int64
        """
        return (lon + 180) % 360 - 180

    @staticmethod
    def validate_coordinate_ranges(lats: pd.Series, lons: pd.Series, raise_error: bool = True) -> dict[str, Any]:
        """
        Валидирует диапазоны координат.

        Проверяет, что широты находятся в диапазоне [-90, 90],
        а долготы - в диапазоне [-180, 180].

        Parameters
        ----------
        lats : pd.Series
            Широты в градусах
        lons : pd.Series
            Долготы в градусах
        raise_error : bool, default=True
            Вызывать ли исключение при обнаружении некорректных значений

        Returns
        -------
        dict[str, Any]
            Словарь с результатами валидации

        Raises
        ------
        ValueError
            Если raise_error=True и обнаружены некорректные координаты
        """
        lat_valid = lats.between(-90, 90).all()
        lon_valid = lons.between(-180, 180).all()

        result = {
            "lat_valid": lat_valid,
            "lon_valid": lon_valid,
            "lat_range": (lats.min(), lats.max()),
            "lon_range": (lons.min(), lons.max()),
            "invalid_lat_count": (~lats.between(-90, 90)).sum(),
            "invalid_lon_count": (~lons.between(-180, 180)).sum(),
        }

        if raise_error and not (lat_valid and lon_valid):
            errors = []
            if not lat_valid:
                errors.append(f"Обнаружены {result['invalid_lat_count']} значений широты вне диапазона [-90, 90]")
            if not lon_valid:
                errors.append(f"Обнаружены {result['invalid_lon_count']} значений долготы вне диапазона [-180, 180]")
            raise ValueError("; ".join(errors))

        return result

    @staticmethod
    def circ_diff(lon_from: pd.Series, lon_to: pd.Series) -> pd.Series:
        """
        Считает разность между долготами, учитывая что они зациклены.

        Находит кратчайший путь между двумя долготами. Например,
        от 170° до -170° будет 20° (а не 340°).

        Параметры:
        ----------
        lon_from : pd.Series
            От какой долготы считаем
        lon_to : pd.Series
            До какой долготы считаем

        Возвращает:
        ----------
        pd.Series
            Разность в градусах от -180 до +180

        Примеры:
        --------
        >>> import pandas as pd
        >>> lon1 = pd.Series([170, -170, 0])
        >>> lon2 = pd.Series([-170, 170, 180])
        >>> CoordinateProcessor.circ_diff(lon1, lon2)
        0     20  # 170° → -170° = 20° (короткий путь)
        1    -20  # -170° → 170° = -20° (короткий путь)
        2    180  # 0° → 180° = 180°
        dtype: int64
        """
        diff = lon_to - lon_from
        return (diff + 180) % 360 - 180
