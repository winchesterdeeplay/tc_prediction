from pydantic import Field
from pydantic_settings import BaseSettings


class APIConfig(BaseSettings):
    """Конфигурация API сервиса."""

    model_path: str = Field(default="weights/model.onnx", description="Путь к ONNX модели")

    pipeline_type: str = Field(default="fast", description="Тип пайплайна: fast, memory, gpu")

    min_history_length: int = Field(default=1, description="Минимальная длина истории")

    max_history_length: int = Field(default=24, description="Максимальная длина истории")

    host: str = Field(default="0.0.0.0", description="Хост для запуска сервера")

    port: int = Field(default=8000, description="Порт для запуска сервера")

    log_level: str = Field(default="INFO", description="Уровень логирования")

    max_batch_size: int = Field(default=1024, description="Максимальный размер батча")

    default_batch_size: int = Field(default=256, description="Размер батча по умолчанию")

    request_timeout: int = Field(default=300, description="Таймаут запроса в секундах")

    class Config:
        env_file = ".env"
        env_prefix = "API_"

    @property
    def sequence_config(self) -> dict:
        """Возвращает конфигурацию последовательностей."""
        return {"min_history_length": self.min_history_length, "max_history_length": self.max_history_length}


config = APIConfig()
