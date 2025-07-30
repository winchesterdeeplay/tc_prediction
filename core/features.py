from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from core.constants import (
    CoordinateLimits,
    ErrorTemplates,
    FeatureType,
    TimeConstants,
    ValidationMessages,
)
from core.coordinates import CoordinateProcessor


@dataclass
class Feature:
    """
    Единая сущность для представления фичи.

    Attributes
    ----------
    name : str
        Название фичи
    feature_type : FeatureType
        Тип фичи
    description : str
        Описание фичи
    required : bool
        Обязательная ли фича
    validation_rules : Optional[Dict]
        Правила валидации для фичи
    compute_function : Optional[Callable]
        Функция для вычисления фичи (для производных фич)
    """

    name: str
    feature_type: FeatureType
    description: str
    required: bool = True
    validation_rules: dict | None = None
    compute_function: Callable | None = None

    def __post_init__(self) -> None:
        """Инициализация правил валидации по умолчанию."""
        if self.validation_rules is None:
            self.validation_rules = {}

    def validate(self, value: Any) -> bool:
        """
        Валидирует значение фичи.

        Parameters
        ----------
        value : Any
            Значение для валидации

        Returns
        -------
        bool
            True если значение корректно
        """
        if not self.required and value is None:
            return True

        # Проверка диапазонов
        if (
            self.validation_rules
            and "min_value" in self.validation_rules
            and value < self.validation_rules["min_value"]
        ):
            return False
        if (
            self.validation_rules
            and "max_value" in self.validation_rules
            and value > self.validation_rules["max_value"]
        ):
            return False

        # Проверка типов
        if (
            self.validation_rules
            and "dtype" in self.validation_rules
            and not isinstance(value, self.validation_rules["dtype"])
        ):
            return False

        return True

    def compute(self, data: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """
        Вычисляет значение фичи.

        Parameters
        ----------
        data : pd.DataFrame
            Исходные данные
        **kwargs
            Дополнительные параметры

        Returns
        -------
        np.ndarray
            Вычисленные значения фичи
        """
        if self.compute_function is None:
            # Для исходных фич просто извлекаем значения
            result: np.ndarray = data[self.name].values
            return result

        computed_result: Any = self.compute_function(data, **kwargs)
        if isinstance(computed_result, np.ndarray):
            return computed_result
        else:
            return np.array(computed_result)


class FeatureRegistry:
    """
    Реестр всех фич в системе.

    Централизованное место для определения всех фич,
    их типов, правил валидации и функций вычисления.
    """

    def __init__(self) -> None:
        """Инициализирует реестр фич."""
        self._features = self._create_features()
        self._feature_indices = self._build_feature_indices()

    def _create_features(self) -> dict[str, Feature]:
        """Создает все фичи системы."""
        features = {}

        # Исходные фичи
        features.update(
            {
                "lat_deg": Feature(
                    name="lat_deg",
                    feature_type=FeatureType.RAW_SEQUENCE,
                    description="Широта центра циклона в градусах (-90 до 90)",
                    validation_rules={
                        "min_value": CoordinateLimits.LAT_MIN,
                        "max_value": CoordinateLimits.LAT_MAX,
                        "dtype": (int, float),
                    },
                ),
                "lon_deg": Feature(
                    name="lon_deg",
                    feature_type=FeatureType.RAW_SEQUENCE,
                    description="Долгота центра циклона в градусах (-180 до 180)",
                    validation_rules={
                        "min_value": CoordinateLimits.LON_MIN,
                        "max_value": CoordinateLimits.LON_MAX,
                        "dtype": (int, float),
                    },
                ),
                "central_pressure_hpa": Feature(
                    name="central_pressure_hpa",
                    feature_type=FeatureType.RAW_SEQUENCE,
                    description="Центральное давление в гектопаскалях",
                    validation_rules={"dtype": (int, float)},
                ),
                "grade": Feature(
                    name="grade",
                    feature_type=FeatureType.RAW_SEQUENCE,
                    description="Категория циклона: 2=TD, 3=TS, 4=STS, 5=TY, 6=ETC, 9=≥TS",
                    validation_rules={"dtype": int},
                ),
            }
        )

        # Производные фичи
        features.update(
            {
                "velocity_kmh": Feature(
                    name="velocity_kmh",
                    feature_type=FeatureType.DERIVED_SEQUENCE,
                    description="Скорость движения циклона в км/ч",
                    compute_function=self._compute_velocity,
                ),
                "bearing_deg": Feature(
                    name="bearing_deg",
                    feature_type=FeatureType.DERIVED_SEQUENCE,
                    description="Направление движения в градусах (0-360)",
                    compute_function=self._compute_bearing,
                ),
                "acceleration_kmh2": Feature(
                    name="acceleration_kmh2",
                    feature_type=FeatureType.DERIVED_SEQUENCE,
                    description="Ускорение движения в км/ч²",
                    compute_function=self._compute_acceleration,
                ),
                "angular_velocity_deg": Feature(
                    name="angular_velocity_deg",
                    feature_type=FeatureType.DERIVED_SEQUENCE,
                    description="Угловая скорость поворота в градусах/ч",
                    compute_function=self._compute_angular_velocity,
                ),
                "pressure_change_hpa": Feature(
                    name="pressure_change_hpa",
                    feature_type=FeatureType.DERIVED_SEQUENCE,
                    description="Изменение давления в гПа/ч",
                    compute_function=self._compute_pressure_change,
                ),
            }
        )

        # Статические фичи
        features.update(
            {
                "target_time_hours": Feature(
                    name="target_time_hours",
                    feature_type=FeatureType.STATIC,
                    description="Горизонт прогноза в часах (6, 12, 24, 48)",
                    validation_rules={"dtype": int},
                ),
                "day_of_year_sin": Feature(
                    name="day_of_year_sin",
                    feature_type=FeatureType.STATIC,
                    description="Синус дня года (циклический признак)",
                    compute_function=self._compute_day_of_year_sin,
                ),
                "day_of_year_cos": Feature(
                    name="day_of_year_cos",
                    feature_type=FeatureType.STATIC,
                    description="Косинус дня года (циклический признак)",
                    compute_function=self._compute_day_of_year_cos,
                ),
                "month_of_year_sin": Feature(
                    name="month_of_year_sin",
                    feature_type=FeatureType.STATIC,
                    description="Синус месяца года (циклический признак)",
                    compute_function=self._compute_month_of_year_sin,
                ),
                "month_of_year_cos": Feature(
                    name="month_of_year_cos",
                    feature_type=FeatureType.STATIC,
                    description="Косинус месяца года (циклический признак)",
                    compute_function=self._compute_month_of_year_cos,
                ),
            }
        )

        # Метаданные
        features.update(
            {
                "analysis_time": Feature(
                    name="analysis_time",
                    feature_type=FeatureType.METADATA,
                    description="Временная метка анализа",
                    required=True,
                ),
                "intl_id": Feature(
                    name="intl_id",
                    feature_type=FeatureType.METADATA,
                    description="Международный идентификатор циклона",
                    required=True,
                ),
                "storm_name": Feature(
                    name="storm_name", feature_type=FeatureType.METADATA, description="Название циклона", required=False
                ),
            }
        )

        # Целевые фичи
        features.update(
            {
                "dlat_target": Feature(
                    name="dlat_target", feature_type=FeatureType.TARGET, description="Целевое изменение широты"
                ),
                "dlon_target": Feature(
                    name="dlon_target", feature_type=FeatureType.TARGET, description="Целевое изменение долготы"
                ),
            }
        )

        return features

    def _build_feature_indices(self) -> dict[str, int]:
        """Строит словарь индексов для производных фич."""
        derived_features = self.get_features_by_type(FeatureType.DERIVED_SEQUENCE)
        return {feature.name: idx for idx, feature in enumerate(derived_features)}

    def get_feature(self, name: str) -> Feature:
        """Получает фичу по имени."""
        if name not in self._features:
            raise ValueError(f"Фича '{name}' не найдена в реестре")
        return self._features[name]

    def get_features_by_type(self, feature_type: FeatureType) -> list[Feature]:
        """Получает все фичи определенного типа."""
        return [f for f in self._features.values() if f.feature_type == feature_type]

    def get_feature_names_by_type(self, feature_type: FeatureType) -> list[str]:
        """Получает названия всех фич определенного типа."""
        return [f.name for f in self.get_features_by_type(feature_type)]

    def get_required_features(self) -> list[Feature]:
        """Получает все обязательные фичи."""
        return [f for f in self._features.values() if f.required]

    def get_required_feature_names(self) -> list[str]:
        """Получает названия всех обязательных фич."""
        return [f.name for f in self.get_required_features()]

    # Функции вычисления для производных фич
    def _compute_velocity(self, data: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """Вычисляет скорость движения."""
        if len(data) < 2:
            return np.zeros(len(data))

        lats = data["lat_deg"].values
        lons = data["lon_deg"].values
        times = data["analysis_time"].values

        time_diffs = self._compute_time_differences(times)
        velocities = CoordinateProcessor.compute_velocity(lats[:-1], lons[:-1], lats[1:], lons[1:], time_diffs)

        result = np.zeros(len(data))
        result[1:] = velocities
        return result

    def _compute_bearing(self, data: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """Вычисляет направление движения."""
        if len(data) < 2:
            return np.zeros(len(data))

        lats = data["lat_deg"].values
        lons = data["lon_deg"].values

        bearings = CoordinateProcessor.compute_bearing(lats[:-1], lons[:-1], lats[1:], lons[1:])

        result = np.zeros(len(data))
        result[1:] = bearings
        return result

    def _compute_acceleration(self, data: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """Вычисляет ускорение движения."""
        if len(data) < 3:
            return np.zeros(len(data))

        velocities = self._compute_velocity(data)
        times = data["analysis_time"].values

        # Для ускорения нужны разности между соседними скоростями и соответствующие интервалы времени
        velocities_prev = velocities[1:-1]  # v1, v2, ..., v_{n-2}
        velocities_next = velocities[2:]  # v2, v3, ..., v_{n-1}
        time_prev = times[1:-1]  # t1, t2, ..., t_{n-2}
        time_next = times[2:]  # t2, t3, ..., t_{n-1}

        delta_v = velocities_next - velocities_prev
        delta_t = (time_next - time_prev).astype("timedelta64[h]").astype(float)
        delta_t = np.where(delta_t == 0, 1.0, delta_t)

        accelerations = delta_v / delta_t

        result = np.zeros(len(data))
        result[2:] = accelerations
        return result

    def _compute_angular_velocity(self, data: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """Вычисляет угловую скорость."""
        if len(data) < 3:
            return np.zeros(len(data))

        bearings = self._compute_bearing(data)
        times = data["analysis_time"].values

        bearings_prev = bearings[1:-1]
        bearings_next = bearings[2:]
        time_prev = times[1:-1]
        time_next = times[2:]

        delta_bearing = bearings_next - bearings_prev
        # Корректируем переходы через 0°/360°
        delta_bearing = np.where(delta_bearing > 180, delta_bearing - 360, delta_bearing)
        delta_bearing = np.where(delta_bearing < -180, delta_bearing + 360, delta_bearing)
        delta_t = (time_next - time_prev).astype("timedelta64[h]").astype(float)
        delta_t = np.where(delta_t == 0, 1.0, delta_t)

        angular_velocities = delta_bearing / delta_t

        result = np.zeros(len(data))
        result[2:] = angular_velocities
        return result

    def _compute_pressure_change(self, data: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """Вычисляет изменение давления."""
        if len(data) < 2:
            return np.zeros(len(data))

        pressures = data["central_pressure_hpa"].values
        times = data["analysis_time"].values

        time_diffs = self._compute_time_differences(times)
        pressure_changes = CoordinateProcessor.compute_pressure_change(pressures[:-1], pressures[1:], time_diffs)

        result = np.zeros(len(data))
        if isinstance(pressure_changes, np.ndarray) and len(pressure_changes) > 0:
            result[1 : 1 + len(pressure_changes)] = pressure_changes
        return result

    def _compute_time_differences(self, times: np.ndarray) -> np.ndarray:
        """Вычисляет разности времени в часах."""
        time_diffs = np.array(
            [(times[i] - times[i - 1]).astype("timedelta64[h]").astype(float) for i in range(1, len(times))]
        )
        return np.where(time_diffs == 0, 1.0, time_diffs)

    # Функции для статических фич
    def _compute_day_of_year_sin(self, data: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """Вычисляет синус дня года."""
        analysis_time = kwargs.get("analysis_time")
        if analysis_time is None:
            return np.zeros(len(data))

        day_of_year = analysis_time.timetuple().tm_yday
        return np.full(len(data), np.sin(TimeConstants.TWO_PI * day_of_year / TimeConstants.DAYS_IN_LEAP_YEAR))

    def _compute_day_of_year_cos(self, data: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """Вычисляет косинус дня года."""
        analysis_time = kwargs.get("analysis_time")
        if analysis_time is None:
            return np.zeros(len(data))

        day_of_year = analysis_time.timetuple().tm_yday
        return np.full(len(data), np.cos(TimeConstants.TWO_PI * day_of_year / TimeConstants.DAYS_IN_LEAP_YEAR))

    def _compute_month_of_year_sin(self, data: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """Вычисляет синус месяца года."""
        analysis_time = kwargs.get("analysis_time")
        if analysis_time is None:
            return np.zeros(len(data))

        month_of_year = analysis_time.month
        return np.full(len(data), np.sin(TimeConstants.TWO_PI * month_of_year / TimeConstants.MONTHS_IN_YEAR))

    def _compute_month_of_year_cos(self, data: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """Вычисляет косинус месяца года."""
        analysis_time = kwargs.get("analysis_time")
        if analysis_time is None:
            return np.zeros(len(data))

        month_of_year = analysis_time.month
        return np.full(len(data), np.cos(TimeConstants.TWO_PI * month_of_year / TimeConstants.MONTHS_IN_YEAR))


class FeatureValidator:
    """
    Валидатор для проверки качества данных и корректности фич.
    """

    def __init__(self, registry: FeatureRegistry) -> None:
        """
        Инициализирует валидатор с реестром фич.

        Parameters
        ----------
        registry : FeatureRegistry
            Реестр фич для валидации
        """
        self.registry = registry

    def _check_missing_columns(self, df: pd.DataFrame, required_columns: set[str], error_message: str) -> None:
        """
        Универсальный метод проверки отсутствующих колонок.

        Parameters
        ----------
        df : pd.DataFrame
            Датафрейм для проверки
        required_columns : Set[str]
            Множество обязательных колонок
        error_message : str
            Сообщение об ошибке
        """
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(ErrorTemplates.MISSING_COLUMNS.format(message=error_message, columns=missing))

    def _check_column_ranges(self, df: pd.DataFrame) -> None:
        """
        Проверяет диапазоны значений для колонок.

        Parameters
        ----------
        df : pd.DataFrame
            Датафрейм для проверки
        """
        for col in df.columns:
            if col in self.registry._features:
                feature = self.registry.get_feature(col)
                if feature.validation_rules and "min_value" in feature.validation_rules:
                    if (
                        not df[col]
                        .between(feature.validation_rules["min_value"], feature.validation_rules["max_value"])
                        .all()
                    ):
                        raise ValueError(
                            ErrorTemplates.COLUMN_OUT_OF_RANGE.format(
                                column=col,
                                min_val=feature.validation_rules["min_value"],
                                max_val=feature.validation_rules["max_value"],
                            )
                        )

    def validate_raw(self, df: pd.DataFrame) -> None:
        """
        Проверяет наличие всех исходных фич в датасете.

        Parameters
        ----------
        df : pd.DataFrame
            Датафрейм для проверки

        Raises
        ------
        ValueError
            Если отсутствуют необходимые исходные фичи
        """
        raw_features = set(self.registry.get_feature_names_by_type(FeatureType.RAW_SEQUENCE))
        self._check_missing_columns(df, raw_features, ValidationMessages.MISSING_RAW_FEATURES)

    def validate_required_columns(self, df: pd.DataFrame) -> None:
        """
        Проверяет наличие всех обязательных колонок в датасете.

        Parameters
        ----------
        df : pd.DataFrame
            Датафрейм для проверки

        Raises
        ------
        ValueError
            Если отсутствуют обязательные колонки
        """
        required_features = set(self.registry.get_required_feature_names())
        self._check_missing_columns(df, required_features, ValidationMessages.MISSING_REQUIRED_COLUMNS)

    def validate_data_quality(self, df: pd.DataFrame) -> None:
        """
        Бросает ValueError, если:
        • нет обязательных колонок
        • координаты за пределами допустимых диапазонов
        """
        self.validate_required_columns(df)
        self._check_column_ranges(df)

    def validate_sequences_format(self, df: pd.DataFrame) -> None:
        """
        Проверяет формат колонки с последовательностями.

        Parameters
        ----------
        df : pd.DataFrame
            Датафрейм с колонкой sequences

        Raises
        ------
        ValueError
            Если формат последовательностей некорректен
        """
        sequences_col = "sequences"

        if sequences_col not in df.columns:
            raise ValueError(ValidationMessages.MISSING_SEQUENCES_COLUMN)

        sequences = df[sequences_col]

        if sequences.empty:
            raise ValueError(ValidationMessages.EMPTY_SEQUENCES_COLUMN)

        first_seq = sequences.iloc[0]
        if not isinstance(first_seq, (list, np.ndarray)):
            raise ValueError(ValidationMessages.INVALID_SEQUENCE_FORMAT)

        if len(first_seq) == 0:
            raise ValueError(ValidationMessages.EMPTY_SEQUENCES)

        # Проверяем размерность фич в последовательности
        expected_dim = len(self.registry.get_feature_names_by_type(FeatureType.RAW_SEQUENCE)) + len(
            self.registry.get_feature_names_by_type(FeatureType.DERIVED_SEQUENCE)
        )

        if len(first_seq[0]) != expected_dim:
            raise ValueError(
                f"{ValidationMessages.INVALID_SEQUENCE_DIMENSION}. "
                f"{ErrorTemplates.EXPECTED_DIMENSION.format(expected=expected_dim, actual=len(first_seq[0]))}"
            )

    def validate_target_values(self, df: pd.DataFrame) -> None:
        """
        Проверяет корректность целевых значений.

        Parameters
        ----------
        df : pd.DataFrame
            Датафрейм с целевыми значениями

        Raises
        ------
        ValueError
            Если целевые значения некорректны
        """
        target_col = "target_time_hours"

        if target_col not in df.columns:
            raise ValueError(ValidationMessages.MISSING_TARGET_COLUMN)

        horizons = df[target_col]

        if (horizons <= 0).any():
            raise ValueError(ValidationMessages.NEGATIVE_HORIZONS)

        if not horizons.dtype in ["int64", "float64"] or (horizons % 1 != 0).any():
            print(f"Предупреждение: {ValidationMessages.NON_INTEGER_HORIZONS}")

    def get_validation_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Возвращает сводку по валидации датасета.

        Parameters
        ----------
        df : pd.DataFrame
            Датафрейм для анализа

        Returns
        -------
        dict[str, Any]
            Сводка с информацией о качестве данных
        """
        required_features = set(self.registry.get_required_feature_names())

        summary: dict[str, Any] = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_columns": list(required_features - set(df.columns)),
            "coordinate_ranges": self._get_coordinate_ranges(df),
            "sequence_info": self._get_sequence_info(df),
        }

        return summary

    def _get_coordinate_ranges(self, df: pd.DataFrame) -> dict[str, dict]:
        """Получает информацию о диапазонах координат."""
        ranges = {}
        coord_features = ["lat_deg", "lon_deg"]

        for col in coord_features:
            if col in df.columns and col in self.registry._features:
                feature = self.registry.get_feature(col)
                if (
                    feature.validation_rules
                    and "min_value" in feature.validation_rules
                    and "max_value" in feature.validation_rules
                ):
                    ranges[col] = {
                        "min": df[col].min(),
                        "max": df[col].max(),
                        "valid": df[col]
                        .between(feature.validation_rules["min_value"], feature.validation_rules["max_value"])
                        .all(),
                    }
        return ranges

    def _get_sequence_info(self, df: pd.DataFrame) -> dict[str, Any]:
        """Получает информацию о последовательностях."""
        if "sequences" not in df.columns:
            return {}

        sequences = df["sequences"]
        if len(sequences) == 0:
            return {"total_sequences": 0}

        lengths = sequences.apply(len)
        return {
            "total_sequences": len(sequences),
            "empty_sequences": (lengths == 0).sum(),
            "avg_length": lengths.mean(),
            "min_length": lengths.min(),
            "max_length": lengths.max(),
        }


class FeatureConfig:
    """
    Централизованная конфигурация и вычисление всех фич для модели прогнозирования траекторий циклонов.

    Этот класс использует FeatureRegistry для управления всеми фичами
    и предоставляет удобный интерфейс для работы с ними.
    """

    def __init__(self) -> None:
        """
        Инициализирует конфигурацию фич и создает валидатор.
        """
        self.registry = FeatureRegistry()
        self.validator = FeatureValidator(self.registry)

    @property
    def raw_sequence_features(self) -> list[str]:
        """Исходные фичи из датасета."""
        return self.registry.get_feature_names_by_type(FeatureType.RAW_SEQUENCE)

    @property
    def derived_sequence_features(self) -> list[str]:
        """Фичи, вычисляемые из исходных данных."""
        return self.registry.get_feature_names_by_type(FeatureType.DERIVED_SEQUENCE)

    @property
    def sequence_features(self) -> list[str]:
        """Все фичи для последовательностей."""
        return self.raw_sequence_features + self.derived_sequence_features

    @property
    def static_features(self) -> list[str]:
        """Статические фичи."""
        return self.registry.get_feature_names_by_type(FeatureType.STATIC)

    @property
    def metadata_features(self) -> list[str]:
        """Метаданные фичи."""
        return self.registry.get_feature_names_by_type(FeatureType.METADATA)

    @property
    def target_features(self) -> list[str]:
        """Целевые фичи."""
        return self.registry.get_feature_names_by_type(FeatureType.TARGET)

    @property
    def coordinate_features(self) -> list[str]:
        """Координатные фичи."""
        return self.registry.get_feature_names_by_type(FeatureType.COORDINATE)

    @property
    def sequences_column(self) -> str:
        """Название колонки с последовательностями."""
        return "sequences"

    @property
    def target_time_column(self) -> str:
        """Название колонки с горизонтом прогноза."""
        return "target_time_hours"

    def get_feature_dimensions(self) -> dict[str, int]:
        """Возвращает размерности всех групп фич."""
        return {
            "raw_sequence": len(self.raw_sequence_features),
            "derived_sequence": len(self.derived_sequence_features),
            "sequence": len(self.sequence_features),
            "static": len(self.static_features),
        }

    def get_required_columns(self) -> list[str]:
        """Возвращает список всех обязательных колонок."""
        return self.registry.get_required_feature_names()

    def compute_derived_sequence_features(self, seq_df: pd.DataFrame) -> np.ndarray:
        """
        Вычисляет производные последовательностные фичи.

        Parameters
        ----------
        seq_df : pd.DataFrame
            Подпоследовательность данных для вычисления фич

        Returns
        -------
        np.ndarray
            Массив с производными фичами
        """
        derived_features = self.registry.get_features_by_type(FeatureType.DERIVED_SEQUENCE)
        n_points = len(seq_df)
        result = np.zeros((n_points, len(derived_features)))

        for i, feature in enumerate(derived_features):
            if feature.compute_function:
                computed_result: np.ndarray = feature.compute(seq_df)
                result[:, i] = computed_result

        return result

    def compute_static_features(self, curr: pd.Series, horizon: int) -> dict[str, float]:
        """
        Вычисляет статические фичи для текущего момента времени.

        Parameters
        ----------
        curr : pd.Series
            Текущая точка траектории с временной меткой
        horizon : int
            Горизонт прогноза в часах

        Returns
        -------
        dict[str, float]
            Словарь с статическими фичами
        """
        static_features = self.registry.get_features_by_type(FeatureType.STATIC)
        result = {}

        for feature in static_features:
            if feature.name == "target_time_hours":
                result[feature.name] = float(horizon)
            elif feature.compute_function:
                # Создаем временный датафрейм для вычисления
                temp_df = pd.DataFrame([curr])
                result[feature.name] = feature.compute(temp_df, analysis_time=curr["analysis_time"])[0]

        return result

    def create_enhanced_sequence(self, seq_df: pd.DataFrame) -> np.ndarray:
        """
        Создает улучшенную последовательность с динамическими фичами.

        Parameters
        ----------
        seq_df : pd.DataFrame
            Подпоследовательность данных для создания фич

        Returns
        -------
        np.ndarray
            Массив с объединенными исходными и производными фичами
        """
        # Извлекаем исходные фичи
        raw_features = seq_df[self.raw_sequence_features].values

        # Вычисляем производные фичи
        derived_features = self.compute_derived_sequence_features(seq_df)

        # Проверяем размерности
        if len(raw_features) == 0 or len(derived_features) == 0:
            # Если один из массивов пустой, возвращаем пустой массив
            return np.array([]).reshape(0, len(self.sequence_features))

        # Объединяем фичи
        seq_features: np.ndarray = np.concatenate([raw_features, derived_features], axis=1)

        return seq_features
