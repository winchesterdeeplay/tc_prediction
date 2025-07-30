import os
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Callable, Any
from functools import wraps
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse

from .models import (
    PredictionRequest,
    PredictionResponse,
    MultipleHorizonsRequest,
    MultipleHorizonsResponse,
    ModelInfoResponse,
    HealthResponse,
)
from .service import CyclonePredictionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

prediction_service: CyclonePredictionService | None = None


def handle_errors(func: Callable) -> Callable:
    """Декоратор для обработки ошибок в эндпоинтах."""

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Ошибка в {func.__name__}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return wrapper


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Управление жизненным циклом приложения."""
    global prediction_service

    # Получение конфигурации из переменных окружения
    model_path = os.getenv("MODEL_PATH", "weights/model.onnx")
    pipeline_type = os.getenv("PIPELINE_TYPE", "fast")

    try:
        prediction_service = CyclonePredictionService(model_path=model_path, pipeline_type=pipeline_type)
        logger.info(f"✅ Сервис предсказания инициализирован с моделью: {model_path}")
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации сервиса: {e}")
        raise

    yield

    logger.info("🔄 Завершение работы сервиса...")


app = FastAPI(
    title="Cyclone Trajectory Prediction API",
    description="API для предсказания траекторий циклонов с использованием машинного обучения",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


def get_prediction_service() -> CyclonePredictionService:
    """Dependency для получения сервиса предсказания."""
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Сервис предсказания не инициализирован")
    return prediction_service


@app.get("/", response_model=dict)
async def root() -> dict:
    """Корневой эндпоинт."""
    return {"message": "Cyclone Trajectory Prediction API", "version": "1.0.0", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse)
async def health_check(service: CyclonePredictionService = Depends(get_prediction_service)) -> HealthResponse:
    """Проверка здоровья сервиса."""
    try:
        is_healthy = service.is_healthy()
        return HealthResponse(
            status="healthy" if is_healthy else "unhealthy", model_loaded=is_healthy, timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"❌ Ошибка проверки здоровья: {e}")
        return HealthResponse(status="unhealthy", model_loaded=False, timestamp=datetime.now())


@app.get("/model/info", response_model=ModelInfoResponse)
@handle_errors
async def get_model_info(service: CyclonePredictionService = Depends(get_prediction_service)) -> ModelInfoResponse:
    """Получение информации о модели."""
    return service.get_model_info()


@app.post("/predict", response_model=PredictionResponse)
@handle_errors
async def predict(
    request: PredictionRequest, service: CyclonePredictionService = Depends(get_prediction_service)
) -> PredictionResponse:
    """
    Предсказание траектории циклона.

    Принимает данные о циклоне и возвращает предсказание его траектории
    на заданный горизонт времени.
    """
    result = service.predict(
        cyclone_data=request.cyclone_data, horizon_hours=request.horizon_hours, batch_size=request.batch_size
    )
    return PredictionResponse(**result)


@app.post("/predict/multiple-horizons", response_model=MultipleHorizonsResponse)
@handle_errors
async def predict_multiple_horizons(
    request: MultipleHorizonsRequest, service: CyclonePredictionService = Depends(get_prediction_service)
) -> MultipleHorizonsResponse:
    """
    Предсказание траектории циклона для нескольких горизонтов.

    Принимает данные о циклоне и возвращает предсказания его траектории
    для всех указанных горизонтов времени.
    """
    result = service.predict_multiple_horizons(
        cyclone_data=request.cyclone_data, horizons=request.horizons, batch_size=request.batch_size
    )
    return MultipleHorizonsResponse(**result)


@app.post("/model/reload")
@handle_errors
async def reload_model(service: CyclonePredictionService = Depends(get_prediction_service)) -> dict:
    """Перезагрузка модели."""
    service.reload_model()
    return {"message": "Модель успешно перезагружена"}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Глобальный обработчик исключений."""
    logger.error(f"❌ Необработанное исключение: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Внутренняя ошибка сервера"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
