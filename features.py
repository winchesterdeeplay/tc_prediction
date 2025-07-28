import re
from dataclasses import dataclass
from datetime import datetime
from typing import Type, TypeVar, cast

import numpy as np
import pandas as pd

from cache import MemoryListCache

T_Feat = TypeVar("T_Feat", bound="Feature")

_CYCLONE_CACHE = MemoryListCache()
_INSTANCE_CACHE: dict[Type["Feature"], "Feature"] = {}


def haversine_distance(
    lat1: np.ndarray | float, lon1: np.ndarray | float, lat2: np.ndarray | float, lon2: np.ndarray | float
) -> np.ndarray | float:
    """Вычисляет расстояние между двумя точками на Земле по формуле гаверсинуса."""
    R = 6371.0  # Радиус Земли в километрах
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def _inst(cls: Type[T_Feat]) -> T_Feat:
    """Возвращает синглтон экземпляра *cls*."""
    if cls not in _INSTANCE_CACHE:
        _INSTANCE_CACHE[cls] = cls()
    return cast(T_Feat, _INSTANCE_CACHE[cls])


@dataclass(slots=True)
class Feature:
    """Базовый класс для всех сгенерированных признаков."""

    fill_value: float = -1.0
    id_column: str = "intl_id"
    time_column: str = "analysis_time"

    @staticmethod
    def camel_to_snake(name: str) -> str:
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    @classmethod
    def name(cls) -> str:
        """Возвращает snake_case имя конкретного класса."""
        return cls.camel_to_snake(cls.__name__)

    def calculate(self, row: pd.Series) -> float | int:
        """Вычисляет признак для *row*, сохраняя историю циклона."""

        if self.id_column not in row:
            return self._calculate_simple(row)

        cyclone_data = self._get_cyclone_data(row[self.id_column], row)
        row_idx = len(cyclone_data) - 1
        return self._calculate_with_history(row, cyclone_data, row_idx)

    @staticmethod
    def reset_cyclone_cache(cyclone_id: str | None = None) -> None:
        """Очищает кэш истории – полностью или по конкретному *cyclone_id*."""

        if cyclone_id is None:
            _CYCLONE_CACHE.clear()
        else:
            _CYCLONE_CACHE.delete(cyclone_id)

    @staticmethod
    def get_cyclone_cache_info() -> dict[str, int]:
        """Возвращает словарь {intl_id: длина истории}."""

        result: dict[str, int] = {}
        for cid in _CYCLONE_CACHE.keys():
            records = _CYCLONE_CACHE.get(cid)
            if records is not None:
                result[cid] = len(records)
        return result

    def _calculate_simple(self, row: pd.Series) -> float | int:
        return self.fill_value

    def _calculate_with_history(self, row: pd.Series, cyclone_data: list[dict], row_idx: int) -> float | int:
        return self._calculate_simple(row)

    def _get_cyclone_data(self, cyclone_id: str, row: pd.Series) -> list[dict]:
        """Обновляет историю циклона и возвращает список записей (отсортирован по времени).

        Предполагаем, что входящие записи идут в хронологическом порядке либо с тем же временем,
        поэтому полный пересортировки не требуется – достаточно проверки последней записи."""

        current_record: dict = row.to_dict()
        current_time = current_record.get(self.time_column)

        records = _CYCLONE_CACHE.get(cyclone_id)
        if records is None:
            # Первая запись
            records = [current_record]
            _CYCLONE_CACHE.set(cyclone_id, records)
            return records

        # Проверяем последнюю запись – самый частый случай
        last_time = records[-1].get(self.time_column)
        if last_time == current_time:
            # Заменяем последнюю запись (keep="last")
            records[-1] = current_record
        elif (last_time is None) or (current_time is None) or (current_time > last_time):
            # Просто добавляем в конец (моновозрастание времени)
            records.append(current_record)
        else:
            # Редкий случай – время "из прошлого". Вставляем отсортированно.
            # Используем бинарный поиск для вставки, чтобы не пересортировать весь список.
            from bisect import bisect_right

            times = [rec.get(self.time_column) or datetime.min for rec in records]
            insert_pos = bisect_right(times, current_time)
            if insert_pos < len(records) and records[insert_pos].get(self.time_column) == current_time:
                # Дубликат времени – заменяем
                records[insert_pos] = current_record
            else:
                records.insert(insert_pos, current_record)

        _CYCLONE_CACHE.set(cyclone_id, records)
        return records

    def _get_shifted_value(self, cyclone_data: list[dict], row_idx: int, column: str, shift: int) -> float | int:
        """Возвращает значение *column* со сдвигом *shift* из истории."""

        # Поддерживаем как список словарей, так и старый DataFrame на случай совместимости
        if isinstance(cyclone_data, list):
            target_idx = row_idx - shift
            if target_idx < 0:
                return self.fill_value

            value = cyclone_data[target_idx].get(column, self.fill_value)
            return value if pd.notna(value) else self.fill_value

        # Старый путь – DataFrame
        if column not in cyclone_data.columns:
            return self.fill_value

        target_idx = row_idx - shift
        if target_idx < 0:
            return self.fill_value

        value = cyclone_data.iat[target_idx, cyclone_data.columns.get_loc(column)]
        return value if pd.notna(value) else self.fill_value


class LatitudePrev(Feature):
    """Широта на один шаг назад."""

    def _calculate_with_history(self, row: pd.Series, cd: list[dict], idx: int) -> float:
        result = self._get_shifted_value(cd, idx, "lat_deg", 1)
        if not isinstance(result, float):
            raise ValueError(f"LatitudePrev: result is not a float: {result}")
        return result


class LongitudePrev(Feature):
    """Долгота на один шаг назад."""

    def _calculate_with_history(self, row: pd.Series, cd: list[dict], idx: int) -> float:
        result = self._get_shifted_value(cd, idx, "lon_deg", 1)
        if not isinstance(result, float):
            raise ValueError(f"LongitudePrev: result is not a float: {result}")
        return result


class LatitudePrev2(Feature):
    """Широта на два шага назад."""

    def _calculate_with_history(self, row: pd.Series, cd: list[dict], idx: int) -> float:
        result = self._get_shifted_value(cd, idx, "lat_deg", 2)
        if not isinstance(result, float):
            raise ValueError(f"LatitudePrev2: result is not a float: {result}")
        return result


class LongitudePrev2(Feature):
    """Долгота на два шага назад."""

    def _calculate_with_history(self, row: pd.Series, cd: list[dict], idx: int) -> float:
        result = self._get_shifted_value(cd, idx, "lon_deg", 2)
        if not isinstance(result, float):
            raise ValueError(f"LongitudePrev2: result is not a float: {result}")
        return result


class LatitudePrev3(Feature):
    """Широта на три шага назад."""

    def _calculate_with_history(self, row: pd.Series, cd: list[dict], idx: int) -> float:
        result = self._get_shifted_value(cd, idx, "lat_deg", 3)
        if not isinstance(result, float):
            raise ValueError(f"LatitudePrev3: result is not a float: {result}")
        return result


class LongitudePrev3(Feature):
    """Долгота на три шага назад."""

    def _calculate_with_history(self, row: pd.Series, cd: list[dict], idx: int) -> float:
        result = self._get_shifted_value(cd, idx, "lon_deg", 3)
        if not isinstance(result, float):
            raise ValueError(f"LongitudePrev3: result is not a float: {result}")
        return result


class _BaseDTPrev(Feature):
    hours_back: int = 0

    def _calculate_with_history(self, row: pd.Series, cd: list[dict], idx: int) -> int:
        prev_time = self._get_shifted_value(cd, idx, self.time_column, self.hours_back)
        if prev_time == self.fill_value:
            return int(self.fill_value)

        return int((row[self.time_column] - prev_time).total_seconds() // 3600)


class DTPrevH(_BaseDTPrev):
    """Разница во времени (ч) до предыдущей записи."""

    hours_back = 1


class DTPrevH2(_BaseDTPrev):
    """Разница во времени (ч) до записи на два шага назад."""

    hours_back = 2


class DTPrevH3(_BaseDTPrev):
    """Разница во времени (ч) до записи на три шага назад."""

    hours_back = 3


class _BaseSpeed(Feature):
    shift_lat: Type[Feature]
    shift_lon: Type[Feature]
    shift_dt: Type[_BaseDTPrev]

    def _calculate_with_history(self, row: pd.Series, cd: list[dict], idx: int) -> float:
        dt_h = _inst(self.shift_dt).calculate(row)
        prev_lat = _inst(self.shift_lat).calculate(row)
        prev_lon = _inst(self.shift_lon).calculate(row)

        if (
            dt_h == _inst(self.shift_dt).fill_value
            or prev_lat == _inst(self.shift_lat).fill_value
            or prev_lon == _inst(self.shift_lon).fill_value
        ):
            return self.fill_value

        current_lat = row.get("lat_deg", self.fill_value)
        current_lon = row.get("lon_deg", self.fill_value)

        if current_lat == self.fill_value or current_lon == self.fill_value:
            return self.fill_value
        distance_km = haversine_distance(prev_lat, prev_lon, current_lat, current_lon)

        return float(distance_km / (dt_h + 1e-6))


class SpeedPrev(_BaseSpeed):
    """Скорость циклона на один шаг назад."""

    fill_value = 25.0
    shift_lat = LatitudePrev
    shift_lon = LongitudePrev
    shift_dt = DTPrevH


class SpeedPrev2(_BaseSpeed):
    """Скорость циклона на два шага назад."""

    fill_value = 25.0
    shift_lat = LatitudePrev2
    shift_lon = LongitudePrev2
    shift_dt = DTPrevH2


class SpeedPrev3(_BaseSpeed):
    """Скорость циклона на три шага назад."""

    fill_value = 25.0
    shift_lat = LatitudePrev3
    shift_lon = LongitudePrev3
    shift_dt = DTPrevH3


class _BaseDirection(Feature):
    shift_lat: Type[Feature]
    shift_lon: Type[Feature]

    def _calculate_with_history(self, row: pd.Series, cd: list[dict], idx: int) -> float:
        dlat = _inst(self.shift_lat).calculate(row)
        dlon = _inst(self.shift_lon).calculate(row)

        if dlat == _inst(self.shift_lat).fill_value or dlon == _inst(self.shift_lon).fill_value:
            return self.fill_value

        return float(np.arctan2(dlon, dlat))


class DirectionPrev(_BaseDirection):
    """Направление циклона на один шаг назад."""

    fill_value = 1.4252529802374745
    shift_lat = LatitudePrev
    shift_lon = LongitudePrev


class DirectionPrev2(_BaseDirection):
    """Направление циклона на два шага назад."""

    fill_value = 1.424033130551433
    shift_lat = LatitudePrev2
    shift_lon = LongitudePrev2


class DirectionPrev3(_BaseDirection):
    """Направление циклона на три шага назад."""

    fill_value = 1.422541501361036
    shift_lat = LatitudePrev3
    shift_lon = LongitudePrev3


class Acceleration(Feature):
    """Ускорение циклона."""

    fill_value = 1.9134408115185586

    def _calculate_with_history(self, row: pd.Series, cd: list[dict], idx: int) -> float:
        v1 = _inst(SpeedPrev).calculate(row)
        v2 = _inst(SpeedPrev2).calculate(row)
        dt = _inst(DTPrevH).calculate(row)

        if v1 == _inst(SpeedPrev).fill_value or v2 == _inst(SpeedPrev2).fill_value or dt == _inst(DTPrevH).fill_value:
            return self.fill_value

        return float((v1 - v2) / (dt + 1e-6))


class Curvature(Feature):
    """Кривизна циклона."""

    fill_value = -0.0005200022887625476

    def _calculate_with_history(self, row: pd.Series, cd: list[dict], idx: int) -> float:
        d1 = _inst(DirectionPrev).calculate(row)
        d2 = _inst(DirectionPrev2).calculate(row)
        dt = _inst(DTPrevH).calculate(row)

        if (
            d1 == _inst(DirectionPrev).fill_value
            or d2 == _inst(DirectionPrev2).fill_value
            or dt == _inst(DTPrevH).fill_value
        ):
            return self.fill_value

        delta = np.arctan2(np.sin(d1 - d2), np.cos(d1 - d2))
        return float(delta / (dt + 1e-6))


class AbsLatitude(Feature):
    """Абсолютная широта циклона."""

    fill_value = 19.0

    def _calculate_simple(self, row: pd.Series) -> float:
        return abs(float(row.get("lat_deg", self.fill_value)))


class AbsLongitude(Feature):
    """Абсолютная долгота циклона."""

    fill_value = 19.0

    def _calculate_simple(self, row: pd.Series) -> float:
        return abs(float(row.get("lon_deg", self.fill_value)))


class DayOfYear(Feature):
    """День года."""

    def _calculate_simple(self, row: pd.Series) -> int:
        ts: datetime | None = row.get("analysis_time")
        return ts.timetuple().tm_yday if ts is not None else 1


class SinDay(Feature):
    """Синус дня года."""

    fill_value = -0.7157953492826642

    def _calculate_simple(self, row: pd.Series) -> float:
        doy = _inst(DayOfYear).calculate(row)
        return float(np.sin(2 * np.pi * doy / 365.25))


class CosDay(Feature):
    """Косинус дня года."""

    fill_value = -0.4159483918249421

    def _calculate_simple(self, row: pd.Series) -> float:
        doy = _inst(DayOfYear).calculate(row)
        return float(np.cos(2 * np.pi * doy / 365.25))


class SpeedPressure(Feature):
    """Скорость циклона умноженная на давление."""

    fill_value = 25000.0  # км/ч * гПа - скорость в км/ч умноженная на типичное давление ~1000 гПа

    def _calculate_with_history(self, row: pd.Series, cd: list[dict], idx: int) -> float:
        cp = row.get("central_pressure_hpa")
        if cp is None or pd.isna(cp):
            return self.fill_value
        return float(_inst(SpeedPrev).calculate(row) * cp)


class LatPressure(Feature):
    """Широта циклона умноженная на давление."""

    fill_value = 17991.4

    def _calculate_with_history(self, row: pd.Series, cd: list[dict], idx: int) -> float:
        cp = row.get("central_pressure_hpa")
        if cp is None or pd.isna(cp):
            return self.fill_value
        return float(_inst(LatitudePrev).calculate(row) * cp)


class Grade(Feature):
    """Класс циклона."""

    fill_value = 4

    def _calculate_simple(self, row: pd.Series) -> int:
        val = row.get("grade", self.fill_value)
        return int(val) if pd.notna(val) else int(self.fill_value)


class MonthOfYear(Feature):
    """Месяц года (1-12) для учета сезонности."""

    fill_value = 8  # пик сезона ураганов

    def _calculate_simple(self, row: pd.Series) -> int:
        ts: datetime | None = row.get("analysis_time")
        return ts.month if ts is not None else self.fill_value


class SinMonth(Feature):
    """Синус месяца для сезонности."""

    fill_value = 0.7071

    def _calculate_simple(self, row: pd.Series) -> float:
        month = MonthOfYear().calculate(row)
        return float(np.sin(2 * np.pi * month / 12))


class CosMonth(Feature):
    """Косинус месяца для сезонности."""

    fill_value = 0.7071

    def _calculate_simple(self, row: pd.Series) -> float:
        month = MonthOfYear().calculate(row)
        return float(np.cos(2 * np.pi * month / 12))


class PressurePrev(Feature):
    """Давление на один шаг назад."""

    def _calculate_with_history(self, row: pd.Series, cd: list[dict], idx: int) -> float:
        result = self._get_shifted_value(cd, idx, "central_pressure_hpa", 1)
        if not isinstance(result, float):
            try:
                result = float(result)
            except (ValueError, TypeError):
                raise ValueError(f"PressurePrev: result is not a float: {result}")
        return result


class PressureChange(Feature):
    """Изменение давления за последний шаг."""

    fill_value = 0.0

    def _calculate_with_history(self, row: pd.Series, cd: list[dict], idx: int) -> float:
        current_pressure = row.get("central_pressure_hpa", self.fill_value)
        prev_pressure = _inst(PressurePrev).calculate(row)

        if (
            current_pressure == self.fill_value
            or prev_pressure == _inst(PressurePrev).fill_value
            or pd.isna(current_pressure)
            or pd.isna(prev_pressure)
        ):
            return self.fill_value

        return float(current_pressure - prev_pressure)


class WindRadius50(Feature):
    """Средний радиус ветров 50+ узлов."""

    fill_value = 60.0

    def _calculate_simple(self, row: pd.Series) -> float:
        long_radius = row.get("r50kt_long_nm", 0)
        short_radius = row.get("r50kt_short_nm", 0)

        if pd.isna(long_radius) or pd.isna(short_radius) or long_radius <= 0 or short_radius <= 0:
            return self.fill_value

        return float((long_radius + short_radius) / 2)


class WindAsymmetry50(Feature):
    """Асимметрия ветров 50+ узлов (отношение длинного к короткому радиусу)."""

    fill_value = 1.5

    def _calculate_simple(self, row: pd.Series) -> float:
        long_radius = row.get("r50kt_long_nm", 0)
        short_radius = row.get("r50kt_short_nm", 0)

        if pd.isna(long_radius) or pd.isna(short_radius) or long_radius <= 0 or short_radius <= 0:
            return self.fill_value

        return float(long_radius / (short_radius + 1e-6))


class DistanceFromEquator(Feature):
    """Расстояние от экватора в градусах."""

    fill_value = 20.0

    def _calculate_simple(self, row: pd.Series) -> float:
        result = AbsLatitude().calculate(row)
        if not isinstance(result, float):
            raise ValueError("DistanceFromEquator: result is not a float")
        return result


class SpeedMean3(Feature):
    """Средняя скорость за последние 3 шага."""

    fill_value = 25.0

    def _calculate_with_history(self, row: pd.Series, cd: list[dict], idx: int) -> float:
        speeds = []
        for i in range(1, 4):
            if idx - i >= 0:
                speed_class = [SpeedPrev, SpeedPrev2, SpeedPrev3][i - 1]
                speed = _inst(speed_class).calculate(row)
                if speed != _inst(speed_class).fill_value:
                    speeds.append(speed)

        return float(np.mean(speeds)) if speeds else self.fill_value


class PressureMean3(Feature):
    """Среднее давление за последние 3 шага."""

    fill_value = 1000.0

    def _calculate_with_history(self, row: pd.Series, cd: list[dict], idx: int) -> float:
        pressures = []
        current_pressure = row.get("central_pressure_hpa")
        if pd.notna(current_pressure):
            pressures.append(current_pressure)

        for i in range(1, 3):
            pressure = self._get_shifted_value(cd, idx, "central_pressure_hpa", i)
            if pressure != self.fill_value and pd.notna(pressure):
                pressures.append(pressure)

        return float(np.mean(pressures)) if pressures else self.fill_value


class SizeSpeedInteraction(Feature):
    """Взаимодействие размера циклона и скорости движения."""

    fill_value = 1500.0

    def _calculate_with_history(self, row: pd.Series, cd: list[dict], idx: int) -> float:
        size = _inst(WindRadius50).calculate(row)
        speed = _inst(SpeedPrev).calculate(row)

        if speed == _inst(SpeedPrev).fill_value:
            speed = 25.0

        return float(size * speed)


class WindRadius30(Feature):
    """Средний радиус ветров 30+ узлов."""

    fill_value = 120.0

    def _calculate_simple(self, row: pd.Series) -> float:
        long_radius = row.get("r30kt_long_nm", 0)
        short_radius = row.get("r30kt_short_nm", 0)

        if pd.isna(long_radius) or pd.isna(short_radius) or long_radius <= 0 or short_radius <= 0:
            return self.fill_value

        return float((long_radius + short_radius) / 2)


class CycloneAge(Feature):
    """Возраст циклона в часах от первого наблюдения."""

    fill_value = 24.0

    def _calculate_with_history(self, row: pd.Series, cd: list[dict], idx: int) -> float:
        if len(cd) <= 1:
            return 0.0

        current_time = row.get("analysis_time")
        if isinstance(cd, list):
            first_time = cd[0].get("analysis_time")
        else:
            first_time = cd.iloc[0].get("analysis_time")

        if current_time is None or first_time is None:
            return self.fill_value

        age_hours = (current_time - first_time).total_seconds() / 3600
        return float(max(0, age_hours))


class PressureTrend3(Feature):
    """Тренд давления за последние 3 шага (наклон линейной регрессии)."""

    fill_value = 0.0

    def _calculate_with_history(self, row: pd.Series, cd: list[dict], idx: int) -> float:
        pressures = []
        times = []

        current_pressure = row.get("central_pressure_hpa")
        current_time = row.get("analysis_time")

        if pd.notna(current_pressure) and current_time is not None:
            pressures.append(current_pressure)
            times.append(0)

        for i in range(1, 4):
            if idx - i >= 0:
                pressure = self._get_shifted_value(cd, idx, "central_pressure_hpa", i)
                if pressure != self.fill_value and pd.notna(pressure):
                    pressures.append(pressure)
                    times.append(-i * 6)  # предполагаем 6-часовые интервалы

        if len(pressures) < 2:
            return self.fill_value

        times = np.array(times)
        pressures = np.array(pressures)

        if np.var(times) == 0:
            return self.fill_value

        slope = np.cov(times, pressures)[0, 1] / np.var(times)
        return float(slope)


class SpeedStd3(Feature):
    """Стандартное отклонение скорости за последние 3 шага."""

    fill_value = 5.0

    def _calculate_with_history(self, row: pd.Series, cd: list[dict], idx: int) -> float:
        speeds = []
        for i in range(1, 4):
            if idx - i >= 0:
                speed_class = [SpeedPrev, SpeedPrev2, SpeedPrev3][i - 1]
                speed = _inst(speed_class).calculate(row)
                if speed != _inst(speed_class).fill_value:
                    speeds.append(speed)

        return float(np.std(speeds)) if len(speeds) >= 2 else self.fill_value


class DirectionChange(Feature):
    """Изменение направления между последними двумя шагами."""

    fill_value = 0.0

    def _calculate_with_history(self, row: pd.Series, cd: list[dict], idx: int) -> float:
        d1 = _inst(DirectionPrev).calculate(row)
        d2 = _inst(DirectionPrev2).calculate(row)

        if d1 == _inst(DirectionPrev).fill_value or d2 == _inst(DirectionPrev2).fill_value:
            return self.fill_value

        # Нормализуем разность углов к [-π, π]
        delta = np.arctan2(np.sin(d1 - d2), np.cos(d1 - d2))
        return float(delta)


class Jerk(Feature):
    """Jerk - производная ускорения (изменение ускорения)."""

    fill_value = 0.0

    def _calculate_with_history(self, row: pd.Series, cd: list[dict], idx: int) -> float:
        if idx < 2:
            return self.fill_value

        v0 = row.get("lat_deg", 0)  # текущая скорость (заглушка)
        v1 = _inst(SpeedPrev).calculate(row)
        v2 = _inst(SpeedPrev2).calculate(row)
        v3 = _inst(SpeedPrev3).calculate(row)

        if (
            v1 == _inst(SpeedPrev).fill_value
            or v2 == _inst(SpeedPrev2).fill_value
            or v3 == _inst(SpeedPrev3).fill_value
        ):
            return self.fill_value

        dt = _inst(DTPrevH).calculate(row)
        if dt == _inst(DTPrevH).fill_value or dt == 0:
            return self.fill_value

        # Простое приближение jerk
        a1 = (v1 - v2) / (dt + 1e-6)  # ускорение 1
        a2 = (v2 - v3) / (dt + 1e-6)  # ускорение 2

        return float((a1 - a2) / (dt + 1e-6))


feature_classes: list[Type[Feature]] = [
    # Базовые лаговые признаки
    LatitudePrev,
    LongitudePrev,
    DTPrevH,
    SpeedPrev,
    DirectionPrev,
    LatitudePrev2,
    LongitudePrev2,
    DTPrevH2,
    SpeedPrev2,
    DirectionPrev2,
    LatitudePrev3,
    LongitudePrev3,
    DTPrevH3,
    SpeedPrev3,
    DirectionPrev3,
    # Динамические признаки
    Acceleration,
    Curvature,
    DirectionChange,
    Jerk,
    # Географические признаки
    AbsLatitude,
    AbsLongitude,
    DistanceFromEquator,
    # Временные признаки
    DayOfYear,
    SinDay,
    CosDay,
    MonthOfYear,
    SinMonth,
    CosMonth,
    # Физические характеристики
    Grade,
    PressurePrev,
    PressureChange,
    PressureTrend3,
    WindRadius50,
    WindRadius30,
    WindAsymmetry50,
    CycloneAge,
    # Скользящие статистики
    SpeedMean3,
    SpeedStd3,
    PressureMean3,
    # Взаимодействия
    SpeedPressure,
    LatPressure,
    SizeSpeedInteraction,
]
