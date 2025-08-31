from typing import Any

import folium
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from tabulate import tabulate

from core.features import FeatureConfig
from evaluation.evaluator import extract_current_coordinates
from inference import ONNXInferencePipeline, ONNXInferencePipelineFactory


def extract_coordinates_from_dataframe(cyclone_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Извлекает координаты широты и долготы из данных циклона.

    Параметры:
    -----------
    cyclone_data : pd.DataFrame
        Датафрейм с данными циклона, содержащий либо столбцы 'lat'/'lon',
        либо столбец 'sequences'

    Возвращает:
    --------
    tuple[np.ndarray, np.ndarray]
        Массивы широт и долгот
    """
    if "lat" in cyclone_data.columns and "lon" in cyclone_data.columns:
        return cyclone_data["lat"].values, cyclone_data["lon"].values
    else:
        # Извлекаем из последовательностей
        lats = []
        lons = []
        for seq in cyclone_data["sequences"]:
            lat, lon = extract_current_coordinates(seq)
            lats.append(lat)
            lons.append(lon)
        return np.array(lats), np.array(lons)


def calculate_error_metrics(
    true_lats: np.ndarray, true_lons: np.ndarray, pred_lats: np.ndarray, pred_lons: np.ndarray
) -> tuple[list[float], list[float], list[float]]:
    """
    Вычисляет метрики ошибок для предсказаний траекторий.

    Параметры:
    -----------
    true_lats : np.ndarray
        Истинные значения широты
    true_lons : np.ndarray
        Истинные значения долготы
    pred_lats : np.ndarray
        Предсказанные значения широты
    pred_lons : np.ndarray
        Предсказанные значения долготы

    Возвращает:
    --------
    tuple[list[float], list[float], list[float]]
        (errors_km, displacements_km, directions)
    """
    errors_km = []
    displacements_km = []
    directions = []

    for i in range(len(true_lats)):
        # Вычисляем ошибку в километрах
        lat_diff = pred_lats[i] - true_lats[i]
        lon_diff = pred_lons[i] - true_lons[i]
        lat_km = lat_diff * 111
        lon_km = lon_diff * 111 * np.cos(np.radians(true_lats[i]))
        error_km = np.sqrt(lat_km**2 + lon_km**2)
        errors_km.append(error_km)

        # Вычисляем смещение
        displacement_km = np.sqrt(
            (pred_lats[i] - true_lats[i]) ** 2 * 111**2
            + (pred_lons[i] - true_lons[i]) ** 2 * 111**2 * np.cos(np.radians(true_lats[i])) ** 2
        )
        displacements_km.append(displacement_km)

        # Вычисляем направление (в градусах)
        if displacement_km > 0:
            direction = np.degrees(np.arctan2(lon_diff, lat_diff))
            if direction < 0:
                direction += 360
            directions.append(direction)
        else:
            directions.append(0)

    return errors_km, displacements_km, directions


def get_error_category(error_km: float) -> tuple[str, str]:
    """
    Получает категорию ошибки и цвет на основе величины ошибки.

    Параметры:
    -----------
    error_km : float
        Ошибка в километрах

    Возвращает:
    --------
    tuple[str, str]
        (категория, цвет)
    """
    if error_km < 50:
        return "Отличное", "green"
    elif error_km < 100:
        return "Хорошее", "orange"
    elif error_km < 200:
        return "Плохое", "red"
    else:
        return "Очень плохое", "darkred"


def create_popup_html(
    point_idx: int,
    true_lat: float,
    true_lon: float,
    pred_lat: float,
    pred_lon: float,
    error_km: float,
    displacement_km: float,
    direction: float,
    pred_delta_lat: float,
    pred_delta_lon: float,
    timestamp_info: str = "",
) -> str:
    """
    Создает HTML-контент для всплывающей информации.

    Параметры:
    -----------
    point_idx : int
        Индекс точки
    true_lat, true_lon : float
        Истинные координаты
    pred_lat, pred_lon : float
        Предсказанные координаты
    error_km : float
        Ошибка в километрах
    displacement_km : float
        Смещение в километрах
    direction : float
        Направление в градусах
    pred_delta_lat, pred_delta_lon : float
        Предсказанные дельты
    timestamp_info : str
        Опциональная информация о времени

    Возвращает:
    --------
    str
        HTML-контент для всплывающего окна
    """
    category, color = get_error_category(error_km)

    return f"""
    <div style="width: 280px; font-family: Arial, sans-serif;">
        <h3 style="color: {color}; margin: 0 0 10px 0;">Точка {point_idx+1}</h3>
        {timestamp_info}
        <div style="background-color: #f8f9fa; padding: 8px; border-radius: 5px; margin: 5px 0;">
            <h4 style="margin: 0 0 5px 0;">Истинные координаты</h4>
            <p style="margin: 2px 0;">🌍 Широта: {true_lat:.4f}°</p>
            <p style="margin: 2px 0;">🌍 Долгота: {true_lon:.4f}°</p>
        </div>
        <div style="background-color: #fff3cd; padding: 8px; border-radius: 5px; margin: 5px 0;">
            <h4 style="margin: 0 0 5px 0;">Предсказанные координаты</h4>
            <p style="margin: 2px 0;">🎯 Широта: {pred_lat:.4f}°</p>
            <p style="margin: 2px 0;">🎯 Долгота: {pred_lon:.4f}°</p>
        </div>
        <div style="background-color: #d1ecf1; padding: 8px; border-radius: 5px; margin: 5px 0;">
            <h4 style="margin: 0 0 5px 0;">Метрики качества</h4>
            <p style="margin: 2px 0;">📏 Ошибка: <strong>{error_km:.1f} км</strong> ({category})</p>
            <p style="margin: 2px 0;">🚀 Смещение: <strong>{displacement_km:.1f} км</strong></p>
            <p style="margin: 2px 0;">🧭 Направление: <strong>{direction:.1f}°</strong></p>
        </div>
        <div style="background-color: #e2e3e5; padding: 8px; border-radius: 5px; margin: 5px 0;">
            <h4 style="margin: 0 0 5px 0;">Предсказанные дельты</h4>
            <p style="margin: 2px 0;">Δlat: {pred_delta_lat:.4f}°</p>
            <p style="margin: 2px 0;">Δlon: {pred_delta_lon:.4f}°</p>
        </div>
    </div>
    """


def plot_trajectory(pipeline: Any, cyclone_data: pd.DataFrame, cyclone_id: str, horizon_hours: int = 24) -> None:
    """
    Визуализирует полную траекторию циклона: истинную и предсказанную.

    Параметры:
    -----------
    pipeline : Any
        ONNX Inference Pipeline с методом predict
    cyclone_data : pd.DataFrame
        Датафрейм с сырыми данными циклонов
    cyclone_id : str
        ID циклона для визуализации
    horizon_hours : int
        Горизонт прогноза в часах (6, 12, 24, 48)
    """
    # Проверяем, что cyclone_id является строкой
    cyclone_id = str(cyclone_id)

    # Фильтруем данные для cyclone_id с безопасной проверкой
    try:
        cyclone_subset = cyclone_data[cyclone_data["intl_id"].astype(str) == cyclone_id].copy()
    except (KeyError, AttributeError) as e:
        print(f"Предупреждение: проблема с фильтрацией по intl_id: {e}")
        print(f"Доступные колонки: {list(cyclone_data.columns)}")
        raise ValueError(f"Не удалось отфильтровать данные для циклона {cyclone_id}")

    if len(cyclone_subset) == 0:
        raise ValueError(f"Циклон с ID {cyclone_id} не найден")

    # Сортируем данные по времени
    cyclone_subset = cyclone_subset.sort_values("analysis_time").reset_index(drop=True)

    # Делаем предсказание для каждой точки отдельно
    true_lats: list[float] = []
    true_lons: list[float] = []
    pred_lats: list[float] = []
    pred_lons: list[float] = []

    for i in range(len(cyclone_subset)):
        # Берем срез данных до текущей точки (включая её)
        current_slice = cyclone_subset.iloc[: i + 1].copy()

        try:
            # Получаем предсказание для текущего среза
            prediction = pipeline.predict(current_slice, horizon_hours=horizon_hours)

            if len(prediction) > 0:
                # Берем последнее предсказание (для текущей точки)
                last_pred = prediction.iloc[-1]

                # Истинные координаты (текущая точка)
                true_lat = last_pred["lat_deg"]
                true_lon = last_pred["lon_deg"]

                # Предсказанные координаты
                pred_lat = last_pred["lat_pred"]
                pred_lon = last_pred["lon_pred"]

                true_lats.append(true_lat)
                true_lons.append(true_lon)
                pred_lats.append(pred_lat)
                pred_lons.append(pred_lon)
            else:
                # Если не удалось получить предсказание, используем текущие координаты
                current_point = cyclone_subset.iloc[i]
                true_lats.append(current_point["lat_deg"])
                true_lons.append(current_point["lon_deg"])
                pred_lats.append(current_point["lat_deg"])
                pred_lons.append(current_point["lon_deg"])

        except Exception as e:
            # Если не удалось получить предсказание, используем текущие координаты
            current_point = cyclone_subset.iloc[i]
            true_lats.append(current_point["lat_deg"])
            true_lons.append(current_point["lon_deg"])
            pred_lats.append(current_point["lat_deg"])
            pred_lons.append(current_point["lon_deg"])

    if len(true_lats) == 0:
        raise ValueError(f"Не удалось получить предсказания для циклона {cyclone_id}")

    # Convert to numpy arrays
    true_lats_array = np.array(true_lats)
    true_lons_array = np.array(true_lons)
    pred_lats_array = np.array(pred_lats)
    pred_lons_array = np.array(pred_lons)

    # Вычисляем метрики
    errors_km, displacements_km, _ = calculate_error_metrics(
        true_lats_array, true_lons_array, pred_lats_array, pred_lons_array
    )

    # Создаем карту
    m = folium.Map(location=[np.mean(true_lats_array), np.mean(true_lons_array)], zoom_start=5)

    # Истинная траектория
    folium.PolyLine(
        list(zip(true_lats_array, true_lons_array)), color="green", weight=3, opacity=0.8, tooltip="Истинная траектория"
    ).add_to(m)

    # Предсказанная траектория
    folium.PolyLine(
        list(zip(pred_lats_array, pred_lons_array)),
        color="red",
        weight=3,
        opacity=0.8,
        tooltip="Предсказанная траектория",
    ).add_to(m)

    # Добавляем точки траектории с информацией
    for i in range(len(true_lats_array)):
        category, color = get_error_category(errors_km[i])

        # Создаем контент всплывающего окна
        popup_html = create_popup_html(
            i,
            true_lats_array[i],
            true_lons_array[i],
            pred_lats_array[i],
            pred_lons_array[i],
            errors_km[i],
            displacements_km[i],
            0,
            0,  # dlat_pred
            0,  # dlon_pred
        )

        # Точка истинной траектории
        folium.CircleMarker(
            location=[true_lats_array[i], true_lons_array[i]],
            radius=6,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Точка {i+1}: {category} ({errors_km[i]:.1f} км)",
        ).add_to(m)

        # Точка предсказанной траектории
        folium.CircleMarker(
            location=[pred_lats_array[i], pred_lons_array[i]],
            radius=4,
            color="red",
            fill=True,
            fillColor="red",
            fillOpacity=0.5,
            tooltip=f"Предсказание {i+1}",
        ).add_to(m)

    # Маркеры начала/конца
    folium.Marker(
        [true_lats_array[0], true_lons_array[0]], popup="Начало траектории", icon=folium.Icon(color="blue", icon="play")
    ).add_to(m)

    folium.Marker(
        [true_lats_array[-1], true_lons_array[-1]],
        popup="Конец (истинный)",
        icon=folium.Icon(color="green", icon="stop"),
    ).add_to(m)

    folium.Marker(
        [pred_lats_array[-1], pred_lons_array[-1]],
        popup="Конец (предсказанный)",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(m)

    # Статистика
    avg_error = np.mean(errors_km)
    max_error = np.max(errors_km)

    stats_html = f"""
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 220px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:11px; padding: 8px; border-radius: 5px;">
        <h4 style="margin: 0 0 8px 0;">📊 Статистика</h4>
        <p style="margin: 2px 0;"><strong>Средняя ошибка:</strong> {avg_error:.1f} км</p>
        <p style="margin: 2px 0;"><strong>Максимальная ошибка:</strong> {max_error:.1f} км</p>
        <p style="margin: 2px 0;"><strong>Количество точек:</strong> {len(errors_km)}</p>
        <p style="margin: 2px 0;"><strong>Горизонт:</strong> {horizon_hours} ч</p>
        <hr style="margin: 6px 0;">
        <p style="margin: 2px 0; font-size: 10px;">🟢 &lt; 50 км | 🟠 50-100 км | 🔴 100-200 км | ⚫ &gt; 200 км</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(stats_html))

    # Легенда
    legend_html = """
    <div style="position: fixed; 
                bottom: 10px; left: 10px; width: 180px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:11px; padding: 8px; border-radius: 5px;">
        <h4 style="margin: 0 0 6px 0;">🎯 Траектория циклона</h4>
        <p style="margin: 2px 0;"><span style="color:green;">━━━</span> Истинная траектория</p>
        <p style="margin: 2px 0;"><span style="color:red;">━━━</span> Предсказанная траектория</p>
        <p style="margin: 2px 0;">🟢 Точки с ошибкой &lt; 50 км</p>
        <p style="margin: 2px 0;">🟠 Точки с ошибкой 50-100 км</p>
        <p style="margin: 2px 0;">🔴 Точки с ошибкой 100-200 км</p>
        <p style="margin: 2px 0;">⚫ Точки с ошибкой &gt; 200 км</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    display(m)


def plot_error_distribution(errors: np.ndarray) -> None:
    """
    Создает гистограмму распределения ошибок.

    Параметры:
    -----------
    errors : np.ndarray
        Массив ошибок в километрах
    """
    # Подготавливаем данные для категорий ошибок
    c0 = int((errors < 50).sum())
    c1 = int(((errors >= 50) & (errors < 100)).sum())
    c2 = int(((errors >= 100) & (errors < 300)).sum())
    c3 = int((errors >= 300).sum())

    html = f"""
    <div style="width: 100%; height: 400px;">
        <canvas id="errorDistChart"></canvas>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        const ctx = document.getElementById('errorDistChart').getContext('2d');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: ['0-50', '50-100', '100-300', '300+'],
                datasets: [{{
                    label: 'Распределение ошибок (км)',
                    data: [{c0}, {c1}, {c2}, {c3}],
                    backgroundColor: [
                        'rgba(75, 192, 192, 0.2)',
                        'rgba(153, 102, 255, 0.2)',
                        'rgba(255, 159, 64, 0.2)',
                        'rgba(255, 99, 132, 0.2)'
                    ],
                    borderColor: [
                        'rgba(75, 192, 192, 1)',
                        'rgba(153, 102, 255, 1)',
                        'rgba(255, 159, 64, 1)',
                        'rgba(255, 99, 132, 1)'
                    ],
                    borderWidth: 1
                }}]
            }},
            options: {{
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
    </script>
    """
    display(HTML(html))


def plot_enhanced_trajectory(
    pipeline: Any, cyclone_data: pd.DataFrame, cyclone_id: str, include_timestamps: bool = True, horizon_hours: int = 24
) -> None:
    """
    Расширенная визуализация траектории с временными метками и детальной информацией.

    Параметры:
    -----------
    pipeline : Any
        ONNX Inference Pipeline с методом predict
    cyclone_data : pd.DataFrame
        Датафрейм с сырыми данными циклонов
    cyclone_id : str
        ID циклона для визуализации
    include_timestamps : bool
        Включать ли временные метки в визуализацию
    horizon_hours : int
        Горизонт прогноза в часах (6, 12, 24, 48)
    """
    # Проверяем, что cyclone_id является строкой
    cyclone_id = str(cyclone_id)

    # Фильтруем данные для cyclone_id с безопасной проверкой
    try:
        cyclone_subset = cyclone_data[cyclone_data["intl_id"].astype(str) == cyclone_id].copy()
    except (KeyError, AttributeError) as e:
        print(f"Предупреждение: проблема с фильтрацией по intl_id: {e}")
        print(f"Доступные колонки: {list(cyclone_data.columns)}")
        raise ValueError(f"Не удалось отфильтровать данные для циклона {cyclone_id}")

    if len(cyclone_subset) == 0:
        raise ValueError(f"Циклон с ID {cyclone_id} не найден")

    # Сортируем данные по времени
    cyclone_subset = cyclone_subset.sort_values("analysis_time").reset_index(drop=True)

    # Делаем предсказание для каждой точки отдельно
    true_lats: list[float] = []
    true_lons: list[float] = []
    pred_lats: list[float] = []
    pred_lons: list[float] = []
    timestamps = []

    for i in range(len(cyclone_subset)):
        # Берем срез данных до текущей точки (включая её)
        current_slice = cyclone_subset.iloc[: i + 1].copy()

        try:
            # Получаем предсказание для текущего среза
            prediction = pipeline.predict(current_slice, horizon_hours=horizon_hours)

            if len(prediction) > 0:
                # Берем последнее предсказание (для текущей точки)
                last_pred = prediction.iloc[-1]

                # Истинные координаты (текущая точка)
                true_lat = last_pred["lat_deg"]
                true_lon = last_pred["lon_deg"]

                # Предсказанные координаты
                pred_lat = last_pred["lat_pred"]
                pred_lon = last_pred["lon_pred"]

                true_lats.append(true_lat)
                true_lons.append(true_lon)
                pred_lats.append(pred_lat)
                pred_lons.append(pred_lon)
                timestamps.append(cyclone_subset.iloc[i]["analysis_time"])
            else:
                # Если не удалось получить предсказание, используем текущие координаты
                current_point = cyclone_subset.iloc[i]
                true_lats.append(current_point["lat_deg"])
                true_lons.append(current_point["lon_deg"])
                pred_lats.append(current_point["lat_deg"])
                pred_lons.append(current_point["lon_deg"])
                timestamps.append(current_point["analysis_time"])

        except Exception as e:
            # Если не удалось получить предсказание, используем текущие координаты
            current_point = cyclone_subset.iloc[i]
            true_lats.append(current_point["lat_deg"])
            true_lons.append(current_point["lon_deg"])
            pred_lats.append(current_point["lat_deg"])
            pred_lons.append(current_point["lon_deg"])
            timestamps.append(current_point["analysis_time"])

    if len(true_lats) == 0:
        raise ValueError(f"Не удалось получить предсказания для циклона {cyclone_id}")

    # Convert to numpy arrays
    true_lats_array = np.array(true_lats)
    true_lons_array = np.array(true_lons)
    pred_lats_array = np.array(pred_lats)
    pred_lons_array = np.array(pred_lons)

    # Вычисляем метрики
    errors_km, displacements_km, directions_deg = calculate_error_metrics(
        true_lats_array, true_lons_array, pred_lats_array, pred_lons_array
    )

    # Создаем карту
    m = folium.Map(location=[np.mean(true_lats_array), np.mean(true_lons_array)], zoom_start=5)

    # Истинная траектория
    folium.PolyLine(
        list(zip(true_lats_array, true_lons_array)), color="green", weight=3, opacity=0.8, tooltip="Истинная траектория"
    ).add_to(m)

    # Предсказанная траектория
    folium.PolyLine(
        list(zip(pred_lats_array, pred_lons_array)),
        color="red",
        weight=3,
        opacity=0.8,
        tooltip="Предсказанная траектория",
    ).add_to(m)

    # Добавляем точки траектории с детальной информацией
    for i in range(len(true_lats_array)):
        category, color = get_error_category(errors_km[i])

        # Формируем информацию о времени
        timestamp_info = ""
        if include_timestamps and i < len(timestamps):
            try:
                timestamp = timestamps[i]
                if isinstance(timestamp, str):
                    timestamp_info = f"<p><strong>Время:</strong> {timestamp}</p>"
                elif hasattr(timestamp, "strftime"):
                    timestamp_info = f"<p><strong>Время:</strong> {timestamp.strftime('%Y-%m-%d %H:%M')}</p>"
            except:
                pass

        # Создаем контент всплывающего окна
        popup_html = create_popup_html(
            i,
            true_lats_array[i],
            true_lons_array[i],
            pred_lats_array[i],
            pred_lons_array[i],
            errors_km[i],
            displacements_km[i],
            directions_deg[i],
            0,  # dlat_pred
            0,  # dlon_pred
            timestamp_info,
        )

        # Точка истинной траектории
        folium.CircleMarker(
            location=[true_lats_array[i], true_lons_array[i]],
            radius=6,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Точка {i+1}: {category} ({errors_km[i]:.1f} км)",
        ).add_to(m)

        # Точка предсказанной траектории
        folium.CircleMarker(
            location=[pred_lats_array[i], pred_lons_array[i]],
            radius=4,
            color="red",
            fill=True,
            fillColor="red",
            fillOpacity=0.5,
            tooltip=f"Предсказание {i+1}",
        ).add_to(m)

    # Маркеры начала/конца
    folium.Marker(
        [true_lats_array[0], true_lons_array[0]], popup="Начало траектории", icon=folium.Icon(color="blue", icon="play")
    ).add_to(m)

    folium.Marker(
        [true_lats_array[-1], true_lons_array[-1]],
        popup="Конец (истинный)",
        icon=folium.Icon(color="green", icon="stop"),
    ).add_to(m)

    folium.Marker(
        [pred_lats_array[-1], pred_lons_array[-1]],
        popup="Конец (предсказанный)",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(m)

    # Статистика
    avg_error = np.mean(errors_km)
    max_error = np.max(errors_km)

    stats_html = f"""
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 220px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:11px; padding: 8px; border-radius: 5px;">
        <h4 style="margin: 0 0 8px 0;">📊 Статистика</h4>
        <p style="margin: 2px 0;"><strong>Средняя ошибка:</strong> {avg_error:.1f} км</p>
        <p style="margin: 2px 0;"><strong>Максимальная ошибка:</strong> {max_error:.1f} км</p>
        <p style="margin: 2px 0;"><strong>Количество точек:</strong> {len(errors_km)}</p>
        <p style="margin: 2px 0;"><strong>Горизонт:</strong> {horizon_hours} ч</p>
        <hr style="margin: 6px 0;">
        <p style="margin: 2px 0; font-size: 10px;">🟢 &lt; 50 км | 🟠 50-100 км | 🔴 100-200 км | ⚫ &gt; 200 км</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(stats_html))
    # Легенда
    legend_html = """
    <div style="position: fixed; 
                bottom: 10px; left: 10px; width: 180px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:11px; padding: 8px; border-radius: 5px;">
        <h4 style="margin: 0 0 6px 0;">🎯 Расширенная траектория</h4>
        <p style="margin: 2px 0;"><span style="color:green;">━━━</span> Истинная траектория</p>
        <p style="margin: 2px 0;"><span style="color:red;">━━━</span> Предсказанная траектория</p>
        <p style="margin: 2px 0;">🟢 Точки с ошибкой &lt; 50 км</p>
        <p style="margin: 2px 0;">🟠 Точки с ошибкой 50-100 км</p>
        <p style="margin: 2px 0;">🔴 Точки с ошибкой 100-200 км</p>
        <p style="margin: 2px 0;">⚫ Точки с ошибкой &gt; 200 км</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    display(m)


def plot_animated_trajectory(
    pipeline: Any, cyclone_data: pd.DataFrame, cyclone_id: str, animation_speed: int = 1000, horizon_hours: int = 24
) -> None:
    """
    Создает анимированную визуализацию траектории с использованием inference pipeline.

    Параметры:
    -----------
    pipeline : Any
        ONNX Inference Pipeline с методом predict
    cyclone_data : pd.DataFrame
        Датафрейм с сырыми данными циклонов
    cyclone_id : str
        ID циклона для визуализации
    animation_speed : int
        Скорость анимации в миллисекундах
    horizon_hours : int
        Горизонт прогноза в часах (6, 12, 24, 48)
    """
    # Проверяем, что cyclone_id является строкой
    cyclone_id = str(cyclone_id)

    # Фильтруем данные для cyclone_id с безопасной проверкой
    try:
        cyclone_subset = cyclone_data[cyclone_data["intl_id"].astype(str) == cyclone_id].copy()
    except (KeyError, AttributeError) as e:
        print(f"Предупреждение: проблема с фильтрацией по intl_id: {e}")
        print(f"Доступные колонки: {list(cyclone_data.columns)}")
        raise ValueError(f"Не удалось отфильтровать данные для циклона {cyclone_id}")

    if len(cyclone_subset) == 0:
        raise ValueError(f"Циклон с ID {cyclone_id} не найден")

    # Сортируем данные по времени
    cyclone_subset = cyclone_subset.sort_values("analysis_time").reset_index(drop=True)

    # Делаем предсказание для каждой точки отдельно
    true_lats: list[float] = []
    true_lons: list[float] = []
    pred_lats: list[float] = []
    pred_lons: list[float] = []

    for i in range(len(cyclone_subset)):
        # Берем срез данных до текущей точки (включая её)
        current_slice = cyclone_subset.iloc[: i + 1].copy()

        try:
            # Получаем предсказание для текущего среза
            prediction = pipeline.predict(current_slice, horizon_hours=horizon_hours)

            if len(prediction) > 0:
                # Берем последнее предсказание (для текущей точки)
                last_pred = prediction.iloc[-1]

                # Истинные координаты (текущая точка)
                true_lat = last_pred["lat_deg"]
                true_lon = last_pred["lon_deg"]

                # Предсказанные координаты
                pred_lat = last_pred["lat_pred"]
                pred_lon = last_pred["lon_pred"]

                true_lats.append(true_lat)
                true_lons.append(true_lon)
                pred_lats.append(pred_lat)
                pred_lons.append(pred_lon)
            else:
                # Если не удалось получить предсказание, используем текущие координаты
                current_point = cyclone_subset.iloc[i]
                true_lats.append(current_point["lat_deg"])
                true_lons.append(current_point["lon_deg"])
                pred_lats.append(current_point["lat_deg"])
                pred_lons.append(current_point["lon_deg"])

        except Exception as e:
            # Если не удалось получить предсказание, используем текущие координаты
            current_point = cyclone_subset.iloc[i]
            true_lats.append(current_point["lat_deg"])
            true_lons.append(current_point["lon_deg"])
            pred_lats.append(current_point["lat_deg"])
            pred_lons.append(current_point["lon_deg"])

    if len(true_lats) == 0:
        raise ValueError(f"Не удалось получить предсказания для циклона {cyclone_id}")

    # Convert to numpy arrays
    true_lats_array = np.array(true_lats)
    true_lons_array = np.array(true_lons)
    pred_lats_array = np.array(pred_lats)
    pred_lons_array = np.array(pred_lons)

    # Вычисляем ошибки
    errors_km, _, _ = calculate_error_metrics(true_lats_array, true_lons_array, pred_lats_array, pred_lons_array)

    # Создаем карту
    from folium import plugins

    m = folium.Map(location=[np.mean(true_lats_array), np.mean(true_lons_array)], zoom_start=5, tiles="OpenStreetMap")

    # Подготавливаем данные траектории для анимации
    true_trajectory = [[lat, lon] for lat, lon in zip(true_lats_array, true_lons_array)]
    pred_trajectory = [[lat, lon] for lat, lon in zip(pred_lats_array, pred_lons_array)]

    # Добавляем анимированные пути
    plugins.AntPath(
        locations=true_trajectory,
        popup="Истинная траектория",
        color="green",
        weight=3,
        opacity=0.8,
        delay=animation_speed,
    ).add_to(m)

    plugins.AntPath(
        locations=pred_trajectory,
        popup="Предсказанная траектория",
        color="red",
        weight=3,
        opacity=0.8,
        delay=animation_speed,
    ).add_to(m)

    # Добавляем статичные точки с информацией
    for i in range(len(true_lats_array)):
        category, color = get_error_category(errors_km[i])

        # Получаем информацию о времени
        timestamp_info = ""
        if i < len(cyclone_subset):
            try:
                timestamp = cyclone_subset.iloc[i]["analysis_time"]
                if isinstance(timestamp, str):
                    timestamp_info = f"<p><strong>Время:</strong> {timestamp}</p>"
                elif hasattr(timestamp, "strftime"):
                    timestamp_info = f"<p><strong>Время:</strong> {timestamp.strftime('%Y-%m-%d %H:%M')}</p>"
            except:
                pass

        # Создаем всплывающее окно
        popup_html = create_popup_html(
            i,
            true_lats_array[i],
            true_lons_array[i],
            pred_lats_array[i],
            pred_lons_array[i],
            errors_km[i],
            0,
            0,
            0,  # dlat_pred
            0,  # dlon_pred
            timestamp_info,
        )

        # Точка истинной траектории
        folium.CircleMarker(
            location=[true_lats_array[i], true_lons_array[i]],
            radius=6,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=2,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"Точка {i+1}: {category} ({errors_km[i]:.1f} км)",
        ).add_to(m)

    # Специальные маркеры
    folium.Marker(
        [true_lats_array[0], true_lons_array[0]],
        popup="🚀 Начало траектории",
        icon=folium.Icon(color="blue", icon="play", prefix="fa"),
    ).add_to(m)

    folium.Marker(
        [true_lats_array[-1], true_lons_array[-1]],
        popup="✅ Конец (истинный)",
        icon=folium.Icon(color="green", icon="stop", prefix="fa"),
    ).add_to(m)

    folium.Marker(
        [pred_lats_array[-1], pred_lons_array[-1]],
        popup="🎯 Конец (предсказанный)",
        icon=folium.Icon(color="red", icon="stop", prefix="fa"),
    ).add_to(m)

    # Статистика
    avg_error = np.mean(errors_km)
    max_error = np.max(errors_km)

    stats_html = f"""
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 220px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:11px; padding: 8px; border-radius: 5px;">
        <h4 style="margin: 0 0 8px 0;">📊 Статистика</h4>
        <p style="margin: 2px 0;"><strong>Средняя ошибка:</strong> {avg_error:.1f} км</p>
        <p style="margin: 2px 0;"><strong>Максимальная ошибка:</strong> {max_error:.1f} км</p>
        <p style="margin: 2px 0;"><strong>Количество точек:</strong> {len(errors_km)}</p>
        <p style="margin: 2px 0;"><strong>Горизонт:</strong> {horizon_hours} ч</p>
        <hr style="margin: 6px 0;">
        <p style="margin: 2px 0; font-size: 10px;">🟢 &lt; 50 км | 🟠 50-100 км | 🔴 100-200 км | ⚫ &gt; 200 км</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(stats_html))

    # Легенда
    legend_html = """
    <div style="position: fixed; 
                bottom: 10px; left: 10px; width: 180px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:11px; padding: 8px; border-radius: 5px;">
        <h4 style="margin: 0 0 6px 0;">🎬 Анимированная траектория</h4>
        <p style="margin: 2px 0;"><span style="color:green;">━━━</span> Истинная траектория</p>
        <p style="margin: 2px 0;"><span style="color:red;">━━━</span> Предсказанная траектория</p>
        <p style="margin: 2px 0;">🟢 Точки с ошибкой &lt; 50 км</p>
        <p style="margin: 2px 0;">🟠 Точки с ошибкой 50-100 км</p>
        <p style="margin: 2px 0;">🔴 Точки с ошибкой 100-200 км</p>
        <p style="margin: 2px 0;">⚫ Точки с ошибкой &gt; 200 км</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    display(m)


def create_inference_pipeline(
    model_path: str, pipeline_type: str = "fast", sequence_config: dict | None = None
) -> ONNXInferencePipeline:
    """
    Создает inference pipeline для визуализации.

    Параметры:
    -----------
    model_path : str
        Путь к ONNX модели
    pipeline_type : str
        Тип pipeline: "fast", "memory_efficient", "gpu"
    sequence_config : dict | None
        Конфигурация последовательностей

    Возвращает:
    --------
    ONNXInferencePipeline
        Готовый к использованию inference pipeline
    """
    if pipeline_type == "fast":
        return ONNXInferencePipelineFactory.create_fast_inference(model_path, sequence_config)
    elif pipeline_type == "memory_efficient":
        return ONNXInferencePipelineFactory.create_memory_efficient(model_path, sequence_config)
    elif pipeline_type == "gpu":
        return ONNXInferencePipelineFactory.create_gpu_inference(model_path, sequence_config)
    else:
        raise ValueError(f"Неизвестный тип pipeline: {pipeline_type}. Доступные: fast, memory_efficient, gpu")


def print_sequence_table(
    sequences: list[np.ndarray] | np.ndarray,
    feature_names: list[str] | None = None,
    sequence_ids: list[str] | None = None,
    max_sequences: int = 5,
    max_features: int = 10,
) -> None:
    """
    Выводит красивую таблицу с признаками и их значениями для последовательностей.

    Параметры:
    -----------
    sequences : list[np.ndarray] | np.ndarray
        Список последовательностей или массив последовательностей
    feature_names : list[str] | None, optional
        Названия признаков. Если None, используются стандартные названия
    sequence_ids : list[str] | None, optional
        Идентификаторы последовательностей для подписей
    max_sequences : int, default=5
        Максимальное количество последовательностей для отображения
    max_features : int, default=10
        Максимальное количество признаков для отображения
    """

    if isinstance(sequences, np.ndarray):
        sequences = [sequences]

    sequences = sequences[:max_sequences]

    if feature_names is None:
        feature_cfg = FeatureConfig()
        feature_names = feature_cfg.sequence_features

    if sequence_ids is None:
        sequence_ids = [f"Seq_{i}" for i in range(len(sequences))]

    for seq_idx, (seq, seq_id) in enumerate(zip(sequences, sequence_ids)):
        print(f"\n🔹 {seq_id}")
        print("-" * 60)

        if len(seq.shape) == 2:
            data_matrix = seq
            n_steps, n_features = data_matrix.shape
        else:
            n_steps = len(seq)
            n_features = max(len(step) for step in seq)
            data_matrix = np.full((n_steps, n_features), np.nan)
            for i, step in enumerate(seq):
                data_matrix[i, : len(step)] = step

        n_features = min(n_features, max_features)
        data_matrix = data_matrix[:, :n_features]
        current_feature_names = feature_names[:n_features]

        df = pd.DataFrame(data_matrix, columns=current_feature_names)
        df.index = [f"Шаг {i+1}" for i in range(n_steps)]

        print(f"Размер: {n_steps} шагов × {n_features} признаков")
        print(tabulate(df, headers="keys", tablefmt="grid", floatfmt=".3f"))
