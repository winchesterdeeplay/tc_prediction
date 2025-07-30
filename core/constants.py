from enum import Enum

import numpy as np


class CoordinateLimits:
    """Константы для координатных ограничений."""

    LAT_MIN = -90.0
    LAT_MAX = 90.0
    LON_MIN = -180.0
    LON_MAX = 180.0


class TimeConstants:
    """Константы для временных вычислений."""

    DAYS_IN_LEAP_YEAR = 366.0
    MONTHS_IN_YEAR = 12.0
    TWO_PI = 2 * np.pi
    HOURS_IN_DAY = 24.0


class ValidationMessages:
    """Сообщения об ошибках валидации."""

    MISSING_RAW_FEATURES = "Отсутствуют необходимые исходные фичи"
    MISSING_REQUIRED_COLUMNS = "Отсутствуют обязательные колонки"
    MISSING_SEQUENCES_COLUMN = "Отсутствует колонка с последовательностями"
    EMPTY_SEQUENCES_COLUMN = "Колонка с последовательностями пуста"
    INVALID_SEQUENCE_FORMAT = "Колонка должна содержать списки или массивы"
    EMPTY_SEQUENCES = "Последовательности не могут быть пустыми"
    INVALID_SEQUENCE_DIMENSION = "Неверная размерность фич в последовательности"
    MISSING_TARGET_COLUMN = "Отсутствует колонка с горизонтом прогноза"
    NEGATIVE_HORIZONS = "Горизонты прогноза должны быть положительными"
    NON_INTEGER_HORIZONS = "Горизонты прогноза должны быть целыми числами (часы)"
    LAT_OUT_OF_RANGE = "lat_deg вне диапазона [-90, 90]"
    LON_OUT_OF_RANGE = "lon_deg вне диапазона [-180, 180]"


class ErrorTemplates:
    """Шаблоны для сообщений об ошибках."""

    COLUMN_OUT_OF_RANGE = "{column} вне диапазона [{min_val}, {max_val}]"
    EXPECTED_DIMENSION = "Ожидается {expected}, получено {actual}"
    MISSING_COLUMNS = "{message}: {columns}"


class FeatureType(Enum):
    """Типы фич для группировки."""

    RAW_SEQUENCE = "raw_sequence"
    DERIVED_SEQUENCE = "derived_sequence"
    STATIC = "static"
    METADATA = "metadata"
    TARGET = "target"
    COORDINATE = "coordinate"
