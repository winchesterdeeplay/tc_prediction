from datetime import datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


class CycloneData(BaseModel):
    intl_id: str = Field(..., description="Уникальный идентификатор циклона")
    storm_name: str = Field(..., description="Название циклона")
    analysis_time: datetime = Field(..., description="Время анализа")
    lat_deg: float = Field(..., ge=-90, le=90, description="Широта в градусах")
    lon_deg: float = Field(..., ge=-180, le=180, description="Долгота в градусах")
    central_pressure_hpa: float = Field(..., description="Центральное давление в гПа")
    grade: int = Field(..., ge=2, le=9, description="Категория циклона (2=TD, 3=TS, 4=STS, 5=TY, 6=ETC, 9=≥TS)")


class Prediction(BaseModel):
    """Модель отдельного предсказания."""

    intl_id: str = Field(..., description="Уникальный идентификатор циклона")
    storm_name: str = Field(..., description="Название циклона")
    analysis_time: str | None = Field(..., description="Время анализа")
    lat_deg: float = Field(..., description="Исходная широта")
    lon_deg: float = Field(..., description="Исходная долгота")
    dlat_pred: float = Field(..., description="Предсказанное изменение широты")
    dlon_pred: float = Field(..., description="Предсказанное изменение долготы")
    lat_pred: float = Field(..., description="Предсказанная широта")
    lon_pred: float = Field(..., description="Предсказанная долгота")


class PredictionRequest(BaseModel):
    cyclone_data: list[CycloneData] = Field(..., description="Данные о циклонах")
    horizon_hours: int = Field(..., ge=6, le=48, description="Горизонт прогноза в часах")
    batch_size: int = Field(default=256, ge=1, le=1024, description="Размер батча для инференса")


class PredictionResponse(BaseModel):
    predictions: list[Prediction] = Field(..., description="Список предсказаний")
    horizon_hours: int = Field(..., description="Горизонт прогноза в часах")
    total_predictions: int = Field(..., description="Общее количество предсказаний")
    processing_time_ms: float = Field(..., description="Время обработки в миллисекундах")

    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "intl_id": "2023001",
                        "storm_name": "TEST_STORM",
                        "analysis_time": "2023-01-01T12:00:00",
                        "lat_deg": 15.5,
                        "lon_deg": 120.3,
                        "dlat_pred": 0.5,
                        "dlon_pred": -1.2,
                        "lat_pred": 16.0,
                        "lon_pred": 119.1,
                    }
                ],
                "horizon_hours": 24,
                "total_predictions": 1,
                "processing_time_ms": 45.2,
            }
        }


class MultipleHorizonsRequest(BaseModel):
    """Модель запроса на предсказание для нескольких горизонтов."""

    cyclone_data: list[CycloneData] = Field(..., description="Данные о циклонах")
    horizons: list[int] = Field(default=[6, 12, 24, 48], description="Список горизонтов прогноза в часах")
    batch_size: int = Field(default=256, ge=1, le=1024, description="Размер батча для инференса")


class MultipleHorizonsResponse(BaseModel):
    """Модель ответа с предсказаниями для нескольких горизонтов."""

    predictions: dict[str, list[Prediction]] = Field(..., description="Предсказания по горизонтам")
    horizons: list[int] = Field(..., description="Список горизонтов прогноза")
    total_predictions: int = Field(..., description="Общее количество предсказаний")
    processing_time_ms: float = Field(..., description="Время обработки в миллисекундах")


class ModelInfoResponse(BaseModel):
    """Модель ответа с информацией о модели."""

    model_path: str = Field(..., description="Путь к модели")
    inputs: list[str] = Field(..., description="Входные узлы модели")
    outputs: list[str] = Field(..., description="Выходные узлы модели")
    providers: list[str] = Field(..., description="Доступные провайдеры")
    feature_dimensions: dict[str, int] = Field(..., description="Размерности фич")


class HealthResponse(BaseModel):
    """Модель ответа для проверки здоровья сервиса."""

    status: str = Field(..., description="Статус сервиса")
    model_loaded: bool = Field(..., description="Загружена ли модель")
    timestamp: datetime = Field(..., description="Время проверки")


def cyclone_data_to_dataframe(cyclone_data: list[CycloneData]) -> pd.DataFrame:
    """Конвертирует список CycloneData в pandas DataFrame."""
    data = []
    for cyclone in cyclone_data:
        data.append(
            {
                "intl_id": cyclone.intl_id,
                "storm_name": cyclone.storm_name,
                "analysis_time": cyclone.analysis_time,
                "lat_deg": cyclone.lat_deg,
                "lon_deg": cyclone.lon_deg,
                "central_pressure_hpa": cyclone.central_pressure_hpa,
                "grade": cyclone.grade,
            }
        )
    return pd.DataFrame(data)


def dataframe_to_predictions(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Конвертирует DataFrame с предсказаниями в список словарей."""
    predictions = []
    for _, row in df.iterrows():
        prediction = {
            "intl_id": row["intl_id"],
            "storm_name": row["storm_name"],
            "analysis_time": row["analysis_time"].isoformat() if pd.notna(row["analysis_time"]) else None,
            "lat_deg": float(row["lat_deg"]),
            "lon_deg": float(row["lon_deg"]),
            "dlat_pred": float(row["dlat_pred"]),
            "dlon_pred": float(row["dlon_pred"]),
            "lat_pred": float(row["lat_pred"]),
            "lon_pred": float(row["lon_pred"]),
        }
        predictions.append(prediction)
    return predictions
