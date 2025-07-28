import random

import folium
import numpy as np
import pandas as pd
from IPython.display import HTML, display

from coordinate_utils import CoordinateProcessor
from features import haversine_distance


class ModelEvaluator:
    """
    Класс для оценки качества модели предсказания траекторий циклонов.
    """

    def __init__(self, coordinate_processor: CoordinateProcessor | None = None):
        """
        Parameters
        ----------
        coordinate_processor : CoordinateProcessor, optional
            Процессор координат. Если None, создается новый.
        """
        self.coord_processor = coordinate_processor or CoordinateProcessor()

    def evaluate_horizon(self, model, X_horizon: pd.DataFrame, y_horizon: pd.DataFrame) -> dict[str, float]:
        """
        Оценивает модель для одного горизонта прогноза.

        Parameters
        ----------
        model : trained model
            Обученная модель с методом predict
        X_horizon : pd.DataFrame
            Признаки для одного горизонта
        y_horizon : pd.DataFrame
            Истинные значения для одного горизонта

        Returns
        -------
        Dict[str, float]
            Словарь с метриками
        """
        if len(X_horizon) == 0:
            return {
                "samples": 0,
                "mean_km": np.nan,
                "median_km": np.nan,
                "max_km": np.nan,
                "p50": np.nan,
                "p100": np.nan,
                "p300": np.nan,
            }

        # Получаем предсказания
        predictions = model.predict(X_horizon)

        # Обрабатываем координаты
        lat_true, lon_true, lat_pred, lon_pred = self.coord_processor.process_predictions(
            X_horizon, y_horizon, predictions
        )

        # Вычисляем ошибки в километрах
        errors_km = haversine_distance(lat_true, lon_true, lat_pred, lon_pred)

        # Убеждаемся, что errors_km это numpy array
        if isinstance(errors_km, np.ndarray):
            # Вычисляем метрики
            return {
                "samples": len(X_horizon),
                "mean_km": float(errors_km.mean()),
                "median_km": float(np.median(errors_km)),
                "max_km": float(errors_km.max()),
                "p50": float((errors_km < 50).mean() * 100),
                "p100": float((errors_km < 100).mean() * 100),
                "p300": float((errors_km < 300).mean() * 100),
            }
        else:
            # Для случая одиночного значения
            return {
                "samples": len(X_horizon),
                "mean_km": float(errors_km),
                "median_km": float(errors_km),
                "max_km": float(errors_km),
                "p50": float(100.0 if errors_km < 50 else 0.0),
                "p100": float(100.0 if errors_km < 100 else 0.0),
                "p300": float(100.0 if errors_km < 300 else 0.0),
            }

    def evaluate_all_horizons(
        self, model, X_test: pd.DataFrame, y_test: pd.DataFrame, horizons: list[int], verbose: bool = True
    ) -> dict[int, dict[str, float]]:
        """
        Оценивает модель на всех горизонтах прогноза.

        Parameters
        ----------
        model : trained model
            Обученная модель
        X_test : pd.DataFrame
            Тестовые признаки
        y_test : pd.DataFrame
            Тестовые таргеты
        horizons : List[int]
            Список горизонтов в часах
        verbose : bool
            Выводить прогресс

        Returns
        -------
        Dict[int, Dict[str, float]]
            Результаты для каждого горизонта
        """
        if verbose:
            print("🧪 ТЕСТИРОВАНИЕ МОДЕЛИ (все ошибки — километры)")
            print("=" * 110)
            print(
                f"{'Горизонт':<10}{'Примеров':<10}{'Средняя (км)':<16}{'Медиана (км)':<16}"
                f"{'Макс. ошибка (км)':<20}{'P@50 км(%)':<12}{'P@100 км(%)':<13}{'P@300 км(%)':<13}"
            )
            print("-" * 110)

        results = {}

        for horizon in horizons:
            # Фильтруем данные для текущего горизонта
            mask = X_test["target_time_hours"] == horizon
            X_h = X_test.loc[mask]
            y_h = y_test.loc[mask]

            # Оцениваем горизонт
            try:
                horizon_results = self.evaluate_horizon(model, X_h, y_h)
                results[horizon] = horizon_results

                if verbose:
                    if horizon_results["samples"] == 0:
                        print(f"{horizon:>2} ч       {'N/A':<10}" * 7)
                    else:
                        print(
                            f"{horizon:>2} ч       {horizon_results['samples']:<10,d}"
                            f"{horizon_results['mean_km']:<16.1f}{horizon_results['median_km']:<16.1f}"
                            f"{horizon_results['max_km']:<20.1f}{horizon_results['p50']:<12.1f}"
                            f"{horizon_results['p100']:<13.1f}{horizon_results['p300']:<13.1f}"
                        )

            except Exception as e:
                if verbose:
                    print(f"❌ Ошибка для горизонта {horizon}ч: {e}")
                results[horizon] = {
                    "samples": 0,
                    "mean_km": np.nan,
                    "median_km": np.nan,
                    "max_km": np.nan,
                    "p50": np.nan,
                    "p100": np.nan,
                    "p300": np.nan,
                }

        return results

    def calculate_summary_metrics(self, results: dict[int, dict[str, float]]) -> dict[str, float]:
        """
        Вычисляет сводные метрики по всем горизонтам.

        Parameters
        ----------
        results : Dict[int, Dict[str, float]]
            Результаты оценки по горизонтам

        Returns
        -------
        Dict[str, float]
            Сводные метрики
        """
        valid_results = [r for r in results.values() if r["samples"] > 0 and not np.isnan(r["mean_km"])]

        if not valid_results:
            return {
                "total_samples": 0,
                "avg_mean_km": np.nan,
                "avg_median_km": np.nan,
                "max_error_km": np.nan,
                "avg_p50": np.nan,
                "avg_p100": np.nan,
                "avg_p300": np.nan,
            }

        # Взвешенные средние по количеству примеров
        total_samples = sum(r["samples"] for r in valid_results)
        weights = np.array([r["samples"] for r in valid_results]) / total_samples

        return {
            "total_samples": total_samples,
            "avg_mean_km": np.average([r["mean_km"] for r in valid_results], weights=weights),
            "avg_median_km": np.average([r["median_km"] for r in valid_results], weights=weights),
            "max_error_km": max(r["max_km"] for r in valid_results),
            "avg_p50": np.average([r["p50"] for r in valid_results], weights=weights),
            "avg_p100": np.average([r["p100"] for r in valid_results], weights=weights),
            "avg_p300": np.average([r["p300"] for r in valid_results], weights=weights),
        }

    def create_cyclone_map(
        self, cyclone_id: str, cyclone_data: pd.DataFrame, model, horizon_hours: int = 6
    ) -> tuple[folium.Map | None, float, float]:
        """
        Создает карту траектории циклона с прогнозами.

        Parameters
        ----------
        cyclone_id : str
            Идентификатор циклона
        cyclone_data : pd.DataFrame
            Данные циклона с координатами и признаками
        model : trained model
            Обученная модель
        horizon_hours : int
            Горизонт прогноза в часах

        Returns
        -------
        Tuple[Optional[folium.Map], float, float]
            Карта, средняя ошибка, максимальная ошибка
        """
        # Фильтруем данные по горизонту
        cyclone_data = cyclone_data[cyclone_data["target_time_hours"] == horizon_hours]
        if len(cyclone_data) == 0:
            print(f"⚠️ Нет данных с горизонтом {horizon_hours}ч для циклона {cyclone_id}")
            return None, 0, 0

        # Получаем прогнозы
        cyclone_name = cyclone_data["storm_name"].iloc[0] if "storm_name" in cyclone_data.columns else "без имени"

        # Определяем признаки для модели
        feature_cols = [
            col
            for col in cyclone_data.columns
            if col not in ["intl_id", "storm_name", "analysis_time", "dlat_target", "dlon_target"]
        ]
        cyclone_X = cyclone_data[feature_cols]
        cyclone_y_pred = model.predict(cyclone_X)

        # Создаем карту
        center_lat = cyclone_data["latitude_prev"].mean()
        center_lon = cyclone_data["longitude_prev"].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

        # Траектории
        real_trajectory = []
        pred_trajectory_starts = []
        pred_trajectory_ends = []
        errors = []

        for i, (idx, row) in enumerate(cyclone_data.iterrows()):
            lat_current = row["latitude_prev"]
            lon_current = row["longitude_prev"]
            real_trajectory.append([lat_current, lon_current])

            # Истинные координаты (текущие + дельты)
            lat_target_real = lat_current + row["dlat_target"]
            lon_target_real = self.coord_processor.circ_add(np.array([lon_current]), np.array([row["dlon_target"]]))[0]

            # Предсказанные координаты
            lat_target_pred = lat_current + cyclone_y_pred[i, 0]
            lon_target_pred = self.coord_processor.circ_add(np.array([lon_current]), np.array([cyclone_y_pred[i, 1]]))[
                0
            ]

            if not (np.isnan(lat_target_real) or np.isnan(lon_target_real)):
                target_point = [lat_target_real, lon_target_real]
                if target_point not in real_trajectory:
                    real_trajectory.append(target_point)

                pred_trajectory_starts.append([lat_current, lon_current])
                pred_trajectory_ends.append([lat_target_pred, lon_target_pred])

                # Ошибка в градусах
                error = np.sqrt((lat_target_real - lat_target_pred) ** 2 + (lon_target_real - lon_target_pred) ** 2)
                errors.append(error)

        # Отрисовка реальной траектории
        if len(real_trajectory) > 1:
            folium.PolyLine(real_trajectory, color="#0066FF", weight=4, opacity=0.9).add_to(m)

        # Отрисовка прогнозов
        for i, (start, end) in enumerate(zip(pred_trajectory_starts, pred_trajectory_ends)):
            error = errors[i] if i < len(errors) else 0
            folium.PolyLine([start, end], color="#FF6B35", weight=2, opacity=0.6, dash_array="5,3").add_to(m)
            folium.CircleMarker(
                location=end,
                radius=4,
                color="#FF4500",
                fill=True,
                fillColor="#FFB366",
                fillOpacity=0.8,
                weight=2,
                popup=f"Прогноз {i+1}: ({end[0]:.2f}, {end[1]:.2f})<br>Ошибка: {error:.3f}°",
            ).add_to(m)

        # Прогнозная траектория
        if len(pred_trajectory_ends) > 1:
            folium.PolyLine(pred_trajectory_ends, color="#FF4500", weight=4, opacity=0.9, dash_array="8,4").add_to(m)

        # Точки реальной траектории
        for i, (lat, lon) in enumerate(real_trajectory[::2]):
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color="#0066FF",
                fill=True,
                fillColor="#87CEEB",
                fillOpacity=0.8,
                weight=2,
                popup=f"Точка {i+1}: ({lat:.2f}, {lon:.2f})",
            ).add_to(m)

        # Маркеры начала и конца
        if real_trajectory:
            folium.Marker(real_trajectory[0], icon=folium.Icon(color="green", icon="play")).add_to(m)
            folium.Marker(real_trajectory[-1], icon=folium.Icon(color="red", icon="stop")).add_to(m)

        # Статистика
        avg_error = np.mean(errors) if errors else 0
        max_error = np.max(errors) if errors else 0

        # Легенда
        legend_html = f"""
        <div style="position: fixed; bottom: 10px; left: 10px; width: 220px;
             background-color: rgba(255, 255, 255, 0.95); border: 1px solid #888;
             z-index: 1000; font-size: 11px; padding: 8px; border-radius: 5px;
             box-shadow: 0 2px 8px rgba(0,0,0,0.2);">
        <div style="font-weight: bold; color: #333; margin-bottom: 4px;">
        🌀 {cyclone_id} - {cyclone_name}</div>
        <div style="font-size: 10px; color: #666;">
        Точек: {len(real_trajectory)} | Ср.ошибка: {avg_error:.2f}°</div>
        <div style="font-size: 10px; color: #888; margin-bottom: 2px;">
        Прогноз: {horizon_hours}-часовой горизонт</div>
        <hr style="margin: 4px 0; border: 0; border-top: 1px solid #ddd;">
        <div style="font-size: 10px; line-height: 1.3;">
        <span style="color: #0066FF; font-weight: bold;">━━━</span> Реальная траектория<br>
        <span style="color: #FF4500; font-weight: bold;">┅┅┅</span> Прогнозная траектория<br>
        <span style="color: #FF6B35; font-weight: bold;">╶╶╶</span> Отрезки прогнозов<br>
        <span style="color: #28A745;">🟢</span> Начало | <span style="color: #DC3545;">🔴</span> Конец
        </div></div>"""
        m.get_root().html.add_child(folium.Element(legend_html))

        return m, avg_error, max_error

    def visualize_cyclones(
        self,
        model,
        test_data: pd.DataFrame,
        horizons: list[int] = [6, 24, 48],
        num_cyclones: int = 5,
        random_seed: int = 42,
        save_maps: bool = True,
        display_in_notebook: bool = True,
    ) -> dict[str, list[dict]]:
        """
        Создает визуализацию траекторий для случайно выбранных циклонов.

        Parameters
        ----------
        model : trained model
            Обученная модель
        test_data : pd.DataFrame
            Тестовые данные с интегрированными признаками, таргетами и метаданными
        horizons : List[int]
            Список горизонтов прогноза в часах
        num_cyclones : int
            Количество циклонов для визуализации
        random_seed : int
            Сид для воспроизводимости
        save_maps : bool
            Сохранять ли карты в HTML файлы
        display_in_notebook : bool
            Отображать ли карты в ноутбуке

        Returns
        -------
        Dict[str, List[Dict]]
            Информация о созданных картах по горизонтам
        """
        # Выбираем случайные циклоны
        random.seed(random_seed)
        unique_cyclones = test_data["intl_id"].unique()
        selected_cyclones = random.sample(list(unique_cyclones), min(num_cyclones, len(unique_cyclones)))
        print(f"🌀 Выбранные циклоны: {selected_cyclones}")

        results = {}

        for h in horizons:
            print(f"\n📍 Горизонт прогноза: {h}ч")
            maps_info = []

            for cyclone_id in selected_cyclones:
                cyclone_data = test_data[test_data["intl_id"] == cyclone_id]
                if len(cyclone_data) < 2:
                    continue

                cyclone_map, avg_error, max_error = self.create_cyclone_map(
                    cyclone_id, cyclone_data, model, horizon_hours=h
                )

                if cyclone_map is None:
                    continue

                map_info = {
                    "id": cyclone_id,
                    "name": cyclone_data["storm_name"].iloc[0] if "storm_name" in cyclone_data.columns else "без имени",
                    "points": len(cyclone_data[cyclone_data["target_time_hours"] == h]),
                    "avg_error": avg_error,
                    "max_error": max_error,
                    "filename": None,
                    "map": cyclone_map,
                }

                if save_maps:
                    filename = f"cyclone_{cyclone_id}_{h}h_map.html"
                    cyclone_map.save(filename)
                    map_info["filename"] = filename

                maps_info.append(map_info)

                print(f"🌀 Циклон {cyclone_id} ({map_info['name']})")
                if save_maps:
                    print(f"   Файл: {map_info['filename']}")
                print(f"   Точек: {map_info['points']}, Ср. ошибка: {avg_error:.3f}°")

                if display_in_notebook:
                    display(HTML(f"<h3>Циклон {cyclone_id} — {map_info['name']} ({h}ч прогноз)</h3>"))
                    display(cyclone_map)

            results[f"{h}h"] = maps_info

            # Сводка по горизонту
            print(f"\n📊 Итоговая статистика ({h}ч прогноз):")
            print(f"Создано карт: {len(maps_info)}")
            avg_errors = [info["avg_error"] for info in maps_info if info["avg_error"] > 0]
            if avg_errors:
                print(f"Средняя ошибка: {np.mean(avg_errors):.3f}°")
                print(f"🔹 Лучший циклон: {min(maps_info, key=lambda x: x['avg_error'])['id']}")
                print(f"🔸 Худший циклон: {max(maps_info, key=lambda x: x['avg_error'])['id']}")
            else:
                print("⚠️ Нет успешно обработанных циклонов.")

        return results
