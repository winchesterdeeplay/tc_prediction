# Cyclone Trajectory Prediction API

Сервис для предсказания траекторий циклонов.


## API Endpoints

### `GET /health`
Проверка здоровья сервиса.

**Пример:**
```bash
curl -X GET http://localhost:8000/health
```

### `POST /predict`
Предсказание траектории циклона.

**Пример:**
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

**Пример ответа:**
```json
{
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
      "lon_pred": 119.1
    },
    {
      "intl_id": "2023001",
      "storm_name": "TEST_STORM",
      "analysis_time": "2023-01-01T18:00:00",
      "lat_deg": 16.2,
      "lon_deg": 119.8,
      "dlat_pred": 0.3,
      "dlon_pred": -0.9,
      "lat_pred": 16.5,
      "lon_pred": 118.9
    },
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
  "total_predictions": 3,
  "processing_time_ms": 78.4
}
```

### `POST /predict/multiple-horizons`
Предсказание траектории циклона для нескольких горизонтов.

**Пример:**
```bash
curl -X POST http://localhost:8000/predict/multiple-horizons \
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
      }
    ],
    "horizons": [6, 12, 24, 48],
    "batch_size": 256
  }'
```

**Пример ответа:**
```json
{
  "predictions": {
    "6": [
      {
        "intl_id": "2023001",
        "storm_name": "TEST_STORM",
        "analysis_time": "2023-01-01T12:00:00",
        "lat_deg": 15.5,
        "lon_deg": 120.3,
        "dlat_pred": 0.2,
        "dlon_pred": -0.8,
        "lat_pred": 15.7,
        "lon_pred": 119.5
      },
      {
        "intl_id": "2023001",
        "storm_name": "TEST_STORM",
        "analysis_time": "2023-01-01T18:00:00",
        "lat_deg": 16.2,
        "lon_deg": 119.8,
        "dlat_pred": 0.1,
        "dlon_pred": -0.6,
        "lat_pred": 16.3,
        "lon_pred": 119.2
      }
    ],
    "12": [
      {
        "intl_id": "2023001",
        "storm_name": "TEST_STORM",
        "analysis_time": "2023-01-01T12:00:00",
        "lat_deg": 15.5,
        "lon_deg": 120.3,
        "dlat_pred": 0.4,
        "dlon_pred": -1.1,
        "lat_pred": 15.9,
        "lon_pred": 119.2
      },
      {
        "intl_id": "2023001",
        "storm_name": "TEST_STORM",
        "analysis_time": "2023-01-01T18:00:00",
        "lat_deg": 16.2,
        "lon_deg": 119.8,
        "dlat_pred": 0.3,
        "dlon_pred": -0.9,
        "lat_pred": 16.5,
        "lon_pred": 118.9
      }
    ],
    "24": [
      {
        "intl_id": "2023001",
        "storm_name": "TEST_STORM",
        "analysis_time": "2023-01-01T12:00:00",
        "lat_deg": 15.5,
        "lon_deg": 120.3,
        "dlat_pred": 0.5,
        "dlon_pred": -1.2,
        "lat_pred": 16.0,
        "lon_pred": 119.1
      },
      {
        "intl_id": "2023001",
        "storm_name": "TEST_STORM",
        "analysis_time": "2023-01-01T18:00:00",
        "lat_deg": 16.2,
        "lon_deg": 119.8,
        "dlat_pred": 0.4,
        "dlon_pred": -1.1,
        "lat_pred": 16.6,
        "lon_pred": 118.7
      }
    ],
    "48": [
      {
        "intl_id": "2023001",
        "storm_name": "TEST_STORM",
        "analysis_time": "2023-01-01T12:00:00",
        "lat_deg": 15.5,
        "lon_deg": 120.3,
        "dlat_pred": 0.8,
        "dlon_pred": -2.1,
        "lat_pred": 16.3,
        "lon_pred": 118.2
      },
      {
        "intl_id": "2023001",
        "storm_name": "TEST_STORM",
        "analysis_time": "2023-01-01T18:00:00",
        "lat_deg": 16.2,
        "lon_deg": 119.8,
        "dlat_pred": 0.7,
        "dlon_pred": -1.9,
        "lat_pred": 16.9,
        "lon_pred": 117.9
      }
    ]
  },
  "horizons": [6, 12, 24, 48],
  "total_predictions": 8,
  "processing_time_ms": 156.8
}
```

## Swagger UI
`http://localhost:8000/docs` 