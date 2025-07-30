import folium
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from tabulate import tabulate  # type: ignore

from core.features import FeatureConfig
from evaluation.evaluator import extract_current_coordinates


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


def plot_trajectory(model, X: pd.DataFrame, y: pd.DataFrame, cyclone_id: str) -> None:
    """
    Визуализирует полную траекторию циклона: истинную и предсказанную.

    Параметры:
    -----------
    model : Any
        Обученная модель с методом predict
    X : pd.DataFrame
        Датафрейм входных признаков
    y : pd.DataFrame
        Датафрейм целевых значений
    cyclone_id : str
        ID циклона для визуализации
    """
    # Фильтруем данные для cyclone_id
    cyclone_data = X[X["intl_id"] == cyclone_id].copy()
    if len(cyclone_data) == 0:
        raise ValueError(f"Циклон с ID {cyclone_id} не найден")

    # Извлекаем координаты
    true_lats, true_lons = extract_coordinates_from_dataframe(cyclone_data)

    # Получаем предсказания
    predictions = model.predict(cyclone_data)
    pred_lats = true_lats + predictions[:, 0]
    pred_lons = true_lons + predictions[:, 1]

    # Вычисляем метрики
    errors_km, displacements_km, _ = calculate_error_metrics(true_lats, true_lons, pred_lats, pred_lons)

    # Создаем карту
    m = folium.Map(location=[np.mean(true_lats), np.mean(true_lons)], zoom_start=5)

    # Истинная траектория
    folium.PolyLine(
        list(zip(true_lats, true_lons)), color="green", weight=3, opacity=0.8, tooltip="Истинная траектория"
    ).add_to(m)

    # Предсказанная траектория
    folium.PolyLine(
        list(zip(pred_lats, pred_lons)), color="red", weight=3, opacity=0.8, tooltip="Предсказанная траектория"
    ).add_to(m)

    # Добавляем точки траектории с информацией
    for i in range(len(true_lats)):
        category, color = get_error_category(errors_km[i])

        # Создаем контент всплывающего окна
        popup_html = create_popup_html(
            i,
            true_lats[i],
            true_lons[i],
            pred_lats[i],
            pred_lons[i],
            errors_km[i],
            displacements_km[i],
            0,
            predictions[i, 0],
            predictions[i, 1],
        )

        # Точка истинной траектории
        folium.CircleMarker(
            location=[true_lats[i], true_lons[i]],
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
            location=[pred_lats[i], pred_lons[i]],
            radius=4,
            color="red",
            fill=True,
            fillColor="red",
            fillOpacity=0.5,
            tooltip=f"Предсказание {i+1}",
        ).add_to(m)

    # Маркеры начала/конца
    folium.Marker(
        [true_lats[0], true_lons[0]], popup="Начало траектории", icon=folium.Icon(color="blue", icon="play")
    ).add_to(m)

    folium.Marker(
        [true_lats[-1], true_lons[-1]], popup="Конец (истинный)", icon=folium.Icon(color="green", icon="stop")
    ).add_to(m)

    folium.Marker(
        [pred_lats[-1], pred_lons[-1]], popup="Конец (предсказанный)", icon=folium.Icon(color="red", icon="stop")
    ).add_to(m)

    # Добавляем легенду
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
        <p><strong>Легенда</strong></p>
        <p><i class="fa fa-circle" style="color:green"></i> Истинная траектория</p>
        <p><i class="fa fa-circle" style="color:red"></i> Предсказанная траектория</p>
        <p><i class="fa fa-circle" style="color:green"></i> Ошибка &lt; 50 км</p>
        <p><i class="fa fa-circle" style="color:orange"></i> Ошибка 50-100 км</p>
        <p><i class="fa fa-circle" style="color:red"></i> Ошибка &gt; 100 км</p>
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
    model, X: pd.DataFrame, y: pd.DataFrame, cyclone_id: str, include_timestamps: bool = True
) -> None:
    """
    Расширенная визуализация траектории с временными метками и детальной информацией.

    Параметры:
    -----------
    model : Any
        Обученная модель с методом predict
    X : pd.DataFrame
        Датафрейм входных признаков
    y : pd.DataFrame
        Датафрейм целевых значений
    cyclone_id : str
        ID циклона для визуализации
    include_timestamps : bool
        Включать ли временные метки в визуализацию
    """
    # Фильтруем данные для cyclone_id
    cyclone_data = X[X["intl_id"] == cyclone_id].copy()
    if len(cyclone_data) == 0:
        raise ValueError(f"Циклон с ID {cyclone_id} не найден")

    # Извлекаем координаты
    true_lats, true_lons = extract_coordinates_from_dataframe(cyclone_data)

    # Получаем предсказания
    predictions = model.predict(cyclone_data)
    pred_lats = true_lats + predictions[:, 0]
    pred_lons = true_lons + predictions[:, 1]

    # Вычисляем метрики
    errors_km, displacements_km, directions = calculate_error_metrics(true_lats, true_lons, pred_lats, pred_lons)

    # Создаем карту
    m = folium.Map(location=[np.mean(true_lats), np.mean(true_lons)], zoom_start=5, tiles="OpenStreetMap")

    # Добавляем траектории
    folium.PolyLine(
        list(zip(true_lats, true_lons)), color="darkgreen", weight=4, opacity=0.9, tooltip="Истинная траектория"
    ).add_to(m)

    folium.PolyLine(
        list(zip(pred_lats, pred_lons)), color="darkred", weight=4, opacity=0.9, tooltip="Предсказанная траектория"
    ).add_to(m)

    # Добавляем точки с детальной информацией
    for i in range(len(true_lats)):
        category, color = get_error_category(errors_km[i])

        # Получаем информацию о времени, если доступна
        timestamp_info = ""
        if include_timestamps and "timestamp" in cyclone_data.columns:
            try:
                timestamp = cyclone_data.iloc[i]["timestamp"]
                if isinstance(timestamp, str):
                    timestamp_info = f"<p><strong>Время:</strong> {timestamp}</p>"
                elif hasattr(timestamp, "strftime"):
                    timestamp_info = f"<p><strong>Время:</strong> {timestamp.strftime('%Y-%m-%d %H:%M')}</p>"
            except:
                pass

        # Создаем детальное всплывающее окно
        popup_html = create_popup_html(
            i,
            true_lats[i],
            true_lons[i],
            pred_lats[i],
            pred_lons[i],
            errors_km[i],
            displacements_km[i],
            directions[i],
            predictions[i, 0],
            predictions[i, 1],
            timestamp_info,
        )

        # Точка истинной траектории
        folium.CircleMarker(
            location=[true_lats[i], true_lons[i]],
            radius=8,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.8,
            weight=2,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Точка {i+1}: {category} ({errors_km[i]:.1f} км)",
        ).add_to(m)

        # Точка предсказания
        folium.CircleMarker(
            location=[pred_lats[i], pred_lons[i]],
            radius=5,
            color="red",
            fill=True,
            fillColor="red",
            fillOpacity=0.6,
            weight=1,
            tooltip=f"Предсказание {i+1}",
        ).add_to(m)

        # Линия ошибки для больших ошибок
        if errors_km[i] > 50:
            folium.PolyLine(
                locations=[[true_lats[i], true_lons[i]], [pred_lats[i], pred_lons[i]]],
                color="gray",
                weight=1,
                opacity=0.5,
                dash_array="5,5",
                tooltip=f"Ошибка: {errors_km[i]:.1f} км",
            ).add_to(m)

    # Специальные маркеры
    folium.Marker(
        [true_lats[0], true_lons[0]],
        popup="🚀 Начало траектории",
        icon=folium.Icon(color="blue", icon="play", prefix="fa"),
    ).add_to(m)

    folium.Marker(
        [true_lats[-1], true_lons[-1]],
        popup="✅ Конец (истинный)",
        icon=folium.Icon(color="green", icon="stop", prefix="fa"),
    ).add_to(m)

    folium.Marker(
        [pred_lats[-1], pred_lons[-1]],
        popup="🎯 Конец (предсказанный)",
        icon=folium.Icon(color="red", icon="stop", prefix="fa"),
    ).add_to(m)

    # Статистика
    avg_error = np.mean(errors_km)
    max_error = np.max(errors_km)
    min_error = np.min(errors_km)

    stats_html = f"""
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 250px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 10px; border-radius: 5px;">
        <h4 style="margin: 0 0 10px 0;">📊 Статистика ошибок</h4>
        <p style="margin: 2px 0;"><strong>Средняя ошибка:</strong> {avg_error:.1f} км</p>
        <p style="margin: 2px 0;"><strong>Максимальная ошибка:</strong> {max_error:.1f} км</p>
        <p style="margin: 2px 0;"><strong>Минимальная ошибка:</strong> {min_error:.1f} км</p>
        <p style="margin: 2px 0;"><strong>Количество точек:</strong> {len(errors_km)}</p>
        <hr style="margin: 8px 0;">
        <p style="margin: 2px 0; font-size: 10px;">Цвета точек:</p>
        <p style="margin: 2px 0; font-size: 10px;">🟢 &lt; 50 км | 🟠 50-100 км | 🔴 100-200 км | ⚫ &gt; 200 км</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(stats_html))

    # Легенда
    legend_html = """
    <div style="position: fixed; 
                bottom: 10px; left: 10px; width: 200px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 10px; border-radius: 5px;">
        <h4 style="margin: 0 0 8px 0;">🗺️ Легенда</h4>
        <p style="margin: 2px 0;"><span style="color:darkgreen;">━━━</span> Истинная траектория</p>
        <p style="margin: 2px 0;"><span style="color:darkred;">━━━</span> Предсказанная траектория</p>
        <p style="margin: 2px 0;">🟢 Точки с ошибкой &lt; 50 км</p>
        <p style="margin: 2px 0;">🟠 Точки с ошибкой 50-100 км</p>
        <p style="margin: 2px 0;">🔴 Точки с ошибкой 100-200 км</p>
        <p style="margin: 2px 0;">⚫ Точки с ошибкой &gt; 200 км</p>
        <p style="margin: 2px 0;">🔴 Малые точки - предсказания</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    display(m)


def plot_animated_trajectory(
    model, X: pd.DataFrame, y: pd.DataFrame, cyclone_id: str, animation_speed: int = 1000
) -> None:
    """
    Создает анимированную визуализацию траектории.

    Параметры:
    -----------
    model : Any
        Обученная модель с методом predict
    X : pd.DataFrame
        Датафрейм входных признаков
    y : pd.DataFrame
        Датафрейм целевых значений
    cyclone_id : str
        ID циклона для визуализации
    animation_speed : int
        Скорость анимации в миллисекундах
    """
    # Фильтруем данные для cyclone_id
    cyclone_data = X[X["intl_id"] == cyclone_id].copy()
    if len(cyclone_data) == 0:
        raise ValueError(f"Циклон с ID {cyclone_id} не найден")

    # Извлекаем координаты
    true_lats, true_lons = extract_coordinates_from_dataframe(cyclone_data)

    # Получаем предсказания
    predictions = model.predict(cyclone_data)
    pred_lats = true_lats + predictions[:, 0]
    pred_lons = true_lons + predictions[:, 1]

    # Вычисляем ошибки
    errors_km, _, _ = calculate_error_metrics(true_lats, true_lons, pred_lats, pred_lons)

    # Создаем карту
    from folium import plugins

    m = folium.Map(location=[np.mean(true_lats), np.mean(true_lons)], zoom_start=5, tiles="OpenStreetMap")

    # Подготавливаем данные траектории для анимации
    true_trajectory = [[lat, lon] for lat, lon in zip(true_lats, true_lons)]
    pred_trajectory = [[lat, lon] for lat, lon in zip(pred_lats, pred_lons)]

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
    for i in range(len(true_lats)):
        category, color = get_error_category(errors_km[i])

        # Получаем информацию о времени
        timestamp_info = ""
        if "timestamp" in cyclone_data.columns:
            try:
                timestamp = cyclone_data.iloc[i]["timestamp"]
                if isinstance(timestamp, str):
                    timestamp_info = f"<p><strong>Время:</strong> {timestamp}</p>"
                elif hasattr(timestamp, "strftime"):
                    timestamp_info = f"<p><strong>Время:</strong> {timestamp.strftime('%Y-%m-%d %H:%M')}</p>"
            except:
                pass

        # Создаем всплывающее окно
        popup_html = create_popup_html(
            i,
            true_lats[i],
            true_lons[i],
            pred_lats[i],
            pred_lons[i],
            errors_km[i],
            0,
            0,
            predictions[i, 0],
            predictions[i, 1],
            timestamp_info,
        )

        # Точка истинной траектории
        folium.CircleMarker(
            location=[true_lats[i], true_lons[i]],
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
        [true_lats[0], true_lons[0]],
        popup="🚀 Начало траектории",
        icon=folium.Icon(color="blue", icon="play", prefix="fa"),
    ).add_to(m)

    folium.Marker(
        [true_lats[-1], true_lons[-1]],
        popup="✅ Конец (истинный)",
        icon=folium.Icon(color="green", icon="stop", prefix="fa"),
    ).add_to(m)

    folium.Marker(
        [pred_lats[-1], pred_lons[-1]],
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
