### Базовое использование

```python
from inference import ONNXInferencePipeline
from data_processing import InferenceDataProcessor, SequenceConfig

# Создание процессора данных для инференса
inference_processor = InferenceDataProcessor(
    sequence_config=SequenceConfig(min_history_length=1, max_history_length=24),
    validate_data=True
)

# Создание pipeline
pipeline = ONNXInferencePipeline("path/to/model.onnx")

# Подготовка данных для инференса
inference_dataset = inference_processor.build_dataset(cyclone_data)

# Предсказание
result = pipeline.predict(
    X=inference_dataset.X,
    horizon_hours=24,
    batch_size=256
)

print(f"Предсказано {len(result)} траекторий")
```



### Использование фабрики

```python
from inference import ONNXInferencePipelineFactory

# Быстрый инференс (многопоточность)
fast_pipeline = ONNXInferencePipelineFactory.create_fast_inference("model.onnx")

# Экономичный по памяти
memory_pipeline = ONNXInferencePipelineFactory.create_memory_efficient("model.onnx")

# GPU инференс (если доступен)
gpu_pipeline = ONNXInferencePipelineFactory.create_gpu_inference("model.onnx")
```

### Предсказания для нескольких горизонтов

```python
from data_processing import InferenceDataProcessor, SequenceConfig

# Создание процессора данных для инференса
inference_processor = InferenceDataProcessor(
    sequence_config=SequenceConfig(min_history_length=6, max_history_length=24),
    validate_data=True
)

# Подготовка данных для инференса
inference_dataset = inference_processor.build_dataset(cyclone_data)

# Предсказания для всех горизонтов
results = pipeline.predict_multiple_horizons(
    X=inference_dataset.X,
    horizons=[6, 12, 24, 48],
    batch_size=256
)

for horizon, result_df in results.items():
    if result_df is not None:
        print(f"Горизонт {horizon}h: {len(result_df)} предсказаний")


```


## 📊 Структура данных

### Входные данные

DataFrame должен содержать следующие колонки:

- `intl_id` - уникальный идентификатор циклона
- `storm_name` - название циклона
- `analysis_time` - время анализа (datetime)
- `lat_deg` - широта в градусах
- `lon_deg` - долгота в градусах
- `central_pressure_hpa` - центральное давление в гПа
- `grade` - категория циклона (2=TD, 3=TS, 4=STS, 5=TY, 6=ETC, 9=≥TS)

### Выходные данные

DataFrame с предсказаниями содержит:

- `dlat_pred` - предсказанное изменение широты
- `dlon_pred` - предсказанное изменение долготы
- `lat_pred` - предсказанная широта
- `lon_pred` - предсказанная долгота
- `lat_deg` - исходная широта
- `lon_deg` - исходная долгота
- Все исходные колонки
