# 🌊 Cyclone Trajectory Prediction

## 🏗️ Архитектура модели


1. **SimpleGRUModel** - базовая архитектура:
   - GRU слои для обработки временных последовательностей
   - Отдельная голова для статических признаков
   - Dropout для регуляризации
   - Выходной слой для предсказания координат

2. **LightningCycloneModel** - расширенная версия с PyTorch Lightning:
   - Автоматическое обучение и валидация
   - Экспорт в ONNX формат

### Входные данные:
- **На сколько часов** (6, 24, 48) вперед делать прогноз 
- **Последовательные признаки**: давление, скорость ветра, температура и др.
- **Статические признаки**: начальные координаты, время года, тип циклона
- **Длины последовательностей**: для обработки переменной длины

### Выходные данные:
- **Изменения координат**: Δlat, Δlon (в градусах)

## 🎯 Формирование таргетов

### Нормализация координат:

1. **Нормализация долгот**:
   ```python
   # Приведение к диапазону [-180, 180]
   normalized_lon = (lon + 180) % 360 - 180
   ```

2. **Вычисление изменений координат**:
   ```python
   # Таргеты - это изменения координат от текущей позиции
   dlat = target_lat - current_lat
   dlon = target_lon - current_lon
   ```


### Дополнительные преобразования:

- **Вычисление расстояний**: формула гаверсинуса для точных географических расстояний
- **Скорости и ускорения**: производные от координат по времени
- **Направления движения**: углы между последовательными точками

## 🚀 API

### Основные эндпоинты:

- `GET /health` - проверка состояния сервиса
- `GET /model/info` - информация о модели
- `POST /predict` - предсказание на один горизонт
- `POST /predict/multiple-horizons` - предсказание на несколько горизонтов

### Пример использования:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cyclone_data": [
      {
        "intl_id": "2023001",
        "storm_name": "TEST_STORM",
        "analysis_time": "2023-01-01T12:00:00",
        "lat_deg": 15.5,
        "lon_deg": 120.3,
        "central_pressure_hpa": 1000.0,
        "grade": 3
      },
      {
        "intl_id": "2023001",
        "storm_name": "TEST_STORM",
        "analysis_time": "2023-01-01T18:00:00",
        "lat_deg": 16.2,
        "lon_deg": 119.8,
        "central_pressure_hpa": 995.0,
        "grade": 4
      },
      {
        "intl_id": "2023001",
        "storm_name": "TEST_STORM",
        "analysis_time": "2023-01-02T00:00:00",
        "lat_deg": 16.8,
        "lon_deg": 119.2,
        "central_pressure_hpa": 990.0,
        "grade": 5
      }
    ],
    "horizon_hours": 24,
    "batch_size": 256
  }'
```

**Ответ:**
```json
{
  "predictions": [
    {
      "intl_id": "2023001",
      "storm_name": "TEST_STORM",
      "analysis_time": "2023-01-02T00:00:00",
      "lat_deg": 16.8,
      "lon_deg": 119.2,
      "dlat_pred": 0.2,
      "dlon_pred": -0.7,
      "lat_pred": 17.0,
      "lon_pred": 118.5
    }
  ],
  "horizon_hours": 24,
  "total_predictions": 1,
  "processing_time_ms": 78.4
}
```

## 🐳 Запуск с Docker

```bash
docker-compose up --build
```


## 📁 Структура проекта

```
tc_prediction/
├── api/                    # FastAPI приложение
│   ├── main.py            # Основные эндпоинты API
│   ├── models.py          # Pydantic модели для запросов/ответов
│   ├── service.py         # Бизнес-логика предсказаний
│   ├── config.py          # Конфигурация API
│   └── README.md          # Документация API
├── core/                   # Основные компоненты
│   ├── coordinates.py     # Работа с географическими координатами
│   ├── features.py        # Конфигурация и обработка признаков
│   └── constants.py       # Константы проекта
├── models/                 # Модели машинного обучения
│   ├── model.py           # Основная архитектура модели
│   ├── losses.py          # Функции потерь
│   └── dataset.py         # Dataset классы для обучения
├── training/               # Обучение модели
│   └── data_module.py     # Загрузка и обработка данных
├── data_processing/        # Обработка сырых данных
│   ├── data_processor.py  # Основной процессор данных
│   ├── base_processor.py  # Базовый класс процессора
│   ├── inference_processor.py # Процессор для инференса
│   ├── dataset_models.py  # Модели данных
│   └── dataset_utils.py   # Утилиты для работы с данными
├── evaluation/            # Оценка качества модели
│   ├── evaluator.py       # Основной класс оценки
│   └── visualization.py   # Визуализация результатов
├── inference/             # Инференс модели
│   ├── onnx_pipeline.py   # ONNX пайплайн для инференса
│   └── README.md          # Документация инференса
├── bst_data/              # Данные BST (Best Track)
│   ├── bst_all.txt        # Сырые данные BST
│   ├── bst_all.csv        # Обработанные данные CSV
│   ├── parse_bst.py       # Парсер BST данных
│   ├── processed_dataset.pkl # Обработанный датасет
│   └── bst_all_description.md # Описание данных
├── weights/               # Сохранённые веса модели
├── train.ipynb           # Jupyter notebook для обучения
├── run_api.py            # Запуск API сервера
├── Dockerfile            # Docker образ
├── docker-compose.yml    # Docker Compose конфигурация
├── .dockerignore         # Исключения для Docker
├── pyproject.toml        # Зависимости проекта
├── uv.lock              # Lock файл зависимостей
├── mypy.ini             # Конфигурация MyPy
├── .python-version      # Версия Python
└── .gitignore           # Исключения Git
```