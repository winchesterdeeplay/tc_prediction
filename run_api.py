import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from api.config import config
from api.service import CyclonePredictionService

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Настройка логирования."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("api.log")],
    )


def validate_model_path(model_path: str) -> bool:
    """Проверка существования модели."""
    if not Path(model_path).exists():
        logger.error(f"Модель не найдена: {model_path}")
        logger.info("Убедитесь, что файл модели существует или укажите правильный путь.")
        return False
    return True


def main() -> None:
    """Основная функция запуска."""
    parser = argparse.ArgumentParser(
        description="Cyclone Trajectory Prediction API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model-path", default=config.model_path, help=f"Путь к ONNX модели (по умолчанию: {config.model_path})"
    )

    parser.add_argument(
        "--pipeline-type",
        choices=["fast", "memory", "gpu"],
        default=config.pipeline_type,
        help=f"Тип пайплайна (по умолчанию: {config.pipeline_type})",
    )

    parser.add_argument("--host", default=config.host, help=f"Хост для запуска сервера (по умолчанию: {config.host})")

    parser.add_argument(
        "--port", type=int, default=config.port, help=f"Порт для запуска сервера (по умолчанию: {config.port})"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=config.log_level,
        help=f"Уровень логирования (по умолчанию: {config.log_level})",
    )

    parser.add_argument(
        "--reload", action="store_true", help="Автоматическая перезагрузка при изменении кода (для разработки)"
    )

    parser.add_argument("--workers", type=int, default=1, help="Количество рабочих процессов (по умолчанию: 1)")

    parser.add_argument("--test", action="store_true", help="Запуск тестового режима (проверка конфигурации и модели)")

    args = parser.parse_args()

    # Настройка логирования
    setup_logging(args.log_level)

    logger.info("🌊 Cyclone Trajectory Prediction API")
    logger.info("=" * 50)

    # Проверка модели
    if not validate_model_path(args.model_path):
        sys.exit(1)

    # Тестовый режим
    if args.test:
        logger.info("🧪 Тестовый режим")
        logger.info("-" * 30)

        try:
            # Проверка конфигурации
            logger.info("✅ Конфигурация корректна")

            # Проверка сервиса
            service = CyclonePredictionService(model_path=args.model_path, pipeline_type=args.pipeline_type)
            logger.info("✅ Сервис инициализирован успешно")

            # Проверка модели
            model_info = service.get_model_info()
            logger.info(f"✅ Модель загружена: {model_info.model_path}")
            logger.info(f"   Входные узлы: {model_info.inputs}")
            logger.info(f"   Выходные узлы: {model_info.outputs}")
            logger.info(f"   Провайдеры: {model_info.providers}")

            logger.info("✅ Все тесты пройдены успешно!")
            return

        except Exception as e:
            logger.error(f"❌ Ошибка в тестовом режиме: {e}")
            sys.exit(1)

    os.environ["MODEL_PATH"] = args.model_path
    os.environ["PIPELINE_TYPE"] = args.pipeline_type

    logger.info(f"📁 Модель: {args.model_path}")
    logger.info(f"⚙️  Пайплайн: {args.pipeline_type}")
    logger.info(f"🌐 Хост: {args.host}")
    logger.info(f"🔌 Порт: {args.port}")
    logger.info(f"📝 Логирование: {args.log_level}")
    logger.info(f"🔄 Автоперезагрузка: {'Да' if args.reload else 'Нет'}")
    logger.info(f"👥 Рабочие процессы: {args.workers}")

    logger.info("🚀 Запуск сервера...")
    logger.info(f"📖 Документация: http://{args.host}:{args.port}/docs")
    logger.info(f"🔍 ReDoc: http://{args.host}:{args.port}/redoc")
    logger.info(f"❤️  Health check: http://{args.host}:{args.port}/health")

    try:
        import uvicorn

        uvicorn.run(
            "api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level=args.log_level.lower(),
            access_log=True,
        )

    except KeyboardInterrupt:
        logger.info("🛑 Сервер остановлен пользователем")
    except Exception as e:
        logger.error(f"❌ Ошибка запуска сервера: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
