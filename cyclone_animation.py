"""
Аниматор траекторий тропических циклонов с визуализацией прогноза.

Примеры использования
---------------------
Запуск из CLI, сохранить видеофайлы в директорию:

```bash
uv run python cyclone_animation.py \
  --data-path bst_data/bst_all.csv \
  --model-path weights/model.onnx \
  --cyclone-ids 2410 2424 \
  --horizons 12 24 48 72 \
  --output-dir cyclone_animations \
  --fps 2.5 --bitrate 1800 --dpi 150
```

Только показать на экране без сохранения:

```bash
uv runpython cyclone_animation.py --cyclone-ids 2410 --horizons 24 --show
```
"""

import argparse
import logging
import warnings
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from tqdm import tqdm

from evaluation.visualization import create_inference_pipeline

warnings.filterwarnings("ignore")

# Настройка логгера
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Цветовая схема для суток (по умолчанию)
DEFAULT_DAY_COLORS = [
    "#FF6B6B",  # День 1: Красный
    "#4ECDC4",  # День 2: Бирюзовый
    "#45B7D1",  # День 3: Синий
    "#96CEB4",  # День 4: Зеленый
    "#FFEAA7",  # День 5: Желтый
    "#DDA0DD",  # День 6: Сливовый
    "#98D8C8",  # День 7: Мятный
    "#F7DC6F",  # День 8: Золотой
    "#BB8FCE",  # День 9: Лавандовый
    "#85C1E9",  # День 10: Голубой
]

# Основные цвета (по умолчанию)
DEFAULT_COLORS = {
    "forecast": "#FF8C00",  # Оранжевый для прогнозов
    "forecast_trail": "#FFB366",  # Светло-оранжевый для следа
    "uncertainty": "#E6E6FA",  # Лавандовый для конуса
    "current": "#FF0000",  # Красный для текущей позиции
    "land": "#F5E6D3",  # Бежевый для суши
    "water": "#E0F6FF",  # Светло-голубой для воды
    "grid": "#D0D0D0",  # Серая сетка
}

# список циклонов из датасета
FIXED_CYCLONES = [(2410, "SHANSHAN", 2024), (2424, "MAN-YI", 2024), (2407, "AMPIL", 2024), (2211, "HINNAMNOR", 2022)]

class CycloneAnimator:
    def __init__(
        self,
        data_path: str,
        model_path: str = "weights/model.onnx",
        *,
        errors_km: dict[int, dict[str, float]] | None = None,
        colors: dict[str, str] | None = None,
        day_colors: list[str] | None = None,
        show_progress: bool = True,
        log_level: int = logging.INFO,
    ):
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.colors = colors or DEFAULT_COLORS
        self.day_colors = day_colors or DEFAULT_DAY_COLORS
        self.show_progress = show_progress

        logging.getLogger().setLevel(log_level)

        self.df = pd.read_csv(self.data_path, low_memory=False)
        self.df["analysis_time"] = pd.to_datetime(self.df["analysis_time"])

        self._setup_model()

        # Ошибки для разных горизонтов (примерные значения)
        self.errors = errors_km or {12: {"p95_km": 150}, 24: {"p95_km": 220}, 48: {"p95_km": 350}, 72: {"p95_km": 480}}

        logger.info(f"Загружено {len(self.df):,} записей")

        # Валидация колонок
        required_columns = {"intl_id", "analysis_time", "lat_deg", "lon_deg", "central_pressure_hpa", "grade"}
        missing = required_columns - set(self.df.columns)
        if missing:
            raise ValueError(f"В датасете отсутствуют обязательные колонки: {sorted(missing)}")

    def _setup_model(self) -> None:
        """Настройка модели."""
        self.pipeline = create_inference_pipeline(str(self.model_path))
        logger.info(f"Модель загружена")

    def _get_day_color(self, hours_from_start: float) -> str:
        """Возвращает цвет для дня."""
        day_index = int(hours_from_start // 24)
        color_index = day_index % len(self.day_colors)
        return self.day_colors[color_index]

    def generate_forecasts(
        self,
        trajectory_full: pd.DataFrame,
        trajectory_display: pd.DataFrame,
        horizon_hours: int,
        *,
        skip_initial_points: int = 1,
    ) -> dict[int, dict[str, float]]:
        """Генерирует прогнозы для заданного горизонта.

        Возвращает словарь по индексу кадра отображаемой траектории.
        """

        logger.info(f"Генерация прогнозов на {horizon_hours}ч...")
        forecasts: dict[int, dict[str, float]] = {}

        indices = range(len(trajectory_display) - 2)
        iterator = tqdm(indices, desc=f"Forecasts {horizon_hours}h") if self.show_progress else indices

        for i in iterator:
            if i < skip_initial_points:
                continue

            try:
                # Находим соответствующий момент времени в полных данных
                current_time = trajectory_display.iloc[i]["analysis_time"]
                full_data_until_current = trajectory_full[trajectory_full["analysis_time"] <= current_time].copy()

                forecast_df = self.pipeline.predict(df=full_data_until_current, horizon_hours=horizon_hours)
                
                current = trajectory_display.iloc[i]
                pred_dlat = float(forecast_df.iloc[-1]["dlat_pred"])
                pred_dlon = float(forecast_df.iloc[-1]["dlon_pred"])

                forecasts[i] = {
                    "current_time": current["analysis_time"],
                    "current_lat": float(current["lat_deg"]),
                    "current_lon": float(current["lon_deg"]),
                    "forecast_lat": float(current["lat_deg"]) + pred_dlat,
                    "forecast_lon": float(current["lon_deg"]) + pred_dlon,
                }
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Пропуск прогноза для i={i}: {e}")
                continue

        return forecasts

    def _draw_earth_map(self, ax: plt.Axes, lat_bounds: tuple[float, float], lon_bounds: tuple[float, float]) -> None:
        """Отрисовка карты с помощью Cartopy."""
        ax.set_xlim(lon_bounds)
        ax.set_ylim(lat_bounds)

        # Добавляем географические элементы
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color="#2F4F4F", alpha=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, color="#696969", alpha=0.6)
        ax.add_feature(cfeature.LAND, facecolor="#F5E6D3", alpha=0.7)  # Бежевый для суши
        ax.add_feature(cfeature.OCEAN, facecolor="#E0F6FF", alpha=0.8)  # Светло-голубой для океана

        # Добавляем крупные озера
        ax.add_feature(cfeature.LAKES, facecolor="#E0F6FF", alpha=0.8)

        # Добавляем сетку координат с подписями
        gl = ax.gridlines(draw_labels=True, alpha=0.3, color="#D0D0D0", linewidth=0.5)
        gl.top_labels = False  # Убираем подписи сверху
        gl.right_labels = False  # Убираем подписи справа
        gl.xlabel_style = {"size": 12, "color": "black"}
        gl.ylabel_style = {"size": 12, "color": "black"}

    def _calculate_equal_aspect_bounds(
        self,
        trajectory: pd.DataFrame,
        forecasts: dict[int, dict[str, float]],
        *,
        margin_percent: float = 0.15,
        min_margin_deg: float = 15,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Вычисляет границы с равным соотношением сторон и гарантией полного захвата."""
        # Получаем границы реальной траектории
        lat_min, lat_max = trajectory["lat_deg"].min(), trajectory["lat_deg"].max()
        lon_min, lon_max = trajectory["lon_deg"].min(), trajectory["lon_deg"].max()

        forecast_lats = [f["forecast_lat"] for f in forecasts.values()]
        forecast_lons = [f["forecast_lon"] for f in forecasts.values()]
        lat_min = min(lat_min, min(forecast_lats))
        lat_max = max(lat_max, max(forecast_lats))
        lon_min = min(lon_min, min(forecast_lons))
        lon_max = max(lon_max, max(forecast_lons))

        # Вычисляем диапазоны
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min

        # Адаптивные отступы: процент от диапазона + фиксированный минимум
        lat_margin = max(min_margin_deg, lat_range * margin_percent)
        lon_margin = max(min_margin_deg, lon_range * margin_percent)

        lat_min, lat_max = lat_min - lat_margin, lat_max + lat_margin
        lon_min, lon_max = lon_min - lon_margin, lon_max + lon_margin

        # Центры областей
        lat_center = (lat_min + lat_max) / 2
        lon_center = (lon_min + lon_max) / 2

        # Размеры областей
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min

        # Коэффициент для равного аспекта (1 градус широты ≈ 111 км)
        # 1 градус долготы = 111 * cos(latitude) км
        cos_lat = np.cos(np.radians(lat_center))

        # Корректируем долготу для равного аспекта
        if lat_range * cos_lat > lon_range:
            # Увеличиваем диапазон долготы
            new_lon_range = lat_range / cos_lat
            lon_min = lon_center - new_lon_range / 2
            lon_max = lon_center + new_lon_range / 2
        else:
            # Увеличиваем диапазон широты
            new_lat_range = lon_range * cos_lat
            lat_min = lat_center - new_lat_range / 2
            lat_max = lat_center + new_lat_range / 2

        return (lat_min, lat_max), (lon_min, lon_max)

    def create_animation(
        self,
        cyclone_id: int,
        horizon_hours: int,
        output_path: str | None = None,
        *,
        sample_every: int = 2,
        skip_initial_points: int = 1,
        margin_percent: float = 0.15,
        min_margin_deg: float = 15,
        interval_ms: int = 400,
        fps: float = 2.5,
        bitrate: int = 1800,
        dpi: int = 150,
        repeat: bool = True,
    ) -> None:
        """Создает анимацию для циклона.

        Если указан `output_path`, файл будет сохранён туда, иначе показан на экране.
        """
        cyclone_name = next((name for cid, name, year in FIXED_CYCLONES if cid == cyclone_id), f"ID_{cyclone_id}")
        logger.info(f"Создание анимации для {cyclone_name} ({cyclone_id}) на {horizon_hours}ч...")

        trajectory = self.df[self.df["intl_id"] == cyclone_id].copy()
        trajectory = trajectory.sort_values("analysis_time").reset_index(drop=True)

        # Сохраняем исходные данные с шагом 6ч для модели
        trajectory_full = trajectory.copy()

        # Фильтруем только для отображения в анимации (пример: 12-часовой шаг при sample_every=2)
        if sample_every > 1:
            trajectory = trajectory.iloc[::sample_every].reset_index(drop=True)

        if trajectory.empty:
            logger.warning(f"Циклон {cyclone_id} не найден")
            return

        logger.info(f"Найдено {len(trajectory)} точек траектории")

        # Генерируем прогнозы используя исходные 6-часовые данные
        forecasts = self.generate_forecasts(
            trajectory_full, trajectory, horizon_hours, skip_initial_points=skip_initial_points
        )

        # Вычисляем границы с равным соотношением сторон (учитывая прогнозы)
        lat_bounds, lon_bounds = self._calculate_equal_aspect_bounds(
            trajectory, forecasts, margin_percent=margin_percent, min_margin_deg=min_margin_deg
        )

        # Оптимизируем размер фигуры под область карты
        lat_range = lat_bounds[1] - lat_bounds[0]
        lon_range = lon_bounds[1] - lon_bounds[0]
        lat_center = (lat_bounds[1] + lat_bounds[0]) / 2
        cos_lat = np.cos(np.radians(lat_center))

        # Используем фиксированные размеры для стабильности ffmpeg
        aspect_ratio = lat_range / (lon_range * cos_lat)

        # Выбираем оптимальный размер из предустановленных (все четные для ffmpeg)
        if aspect_ratio > 1.2:
            fig_width, fig_height = 14, 16  # Вертикальная ориентация
        elif aspect_ratio < 0.8:
            fig_width, fig_height = 18, 12  # Горизонтальная ориентация
        else:
            fig_width, fig_height = 16, 14  # Квадратная ориентация

        # Настройка фигуры с компактной компоновкой
        fig = plt.figure(figsize=(fig_width, fig_height))
        fig.patch.set_facecolor("white")

        # Разделение на области: метаданные + карта + легенда (с большими отступами)
        gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 8, 1.2], hspace=0.2)

        # Область для метаданных (верх)
        ax_meta = fig.add_subplot(gs[0])
        ax_meta.axis("off")

        # Главная область карты с проекцией Cartopy
        ax_map = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())

        # Область для легенды (низ)
        ax_legend = fig.add_subplot(gs[2])
        ax_legend.axis("off")

        storm_name = cyclone_name
        start_time = trajectory.iloc[0]["analysis_time"]

        all_forecasts = []

        def animate(frame: int) -> None:
            ax_map.clear()
            ax_meta.clear()
            ax_legend.clear()

            # Убираем все элементы осей для метаданных и легенды
            ax_meta.axis("off")
            ax_meta.set_xticks([])
            ax_meta.set_yticks([])
            ax_meta.set_frame_on(False)

            ax_legend.axis("off")
            ax_legend.set_xticks([])
            ax_legend.set_yticks([])
            ax_legend.set_frame_on(False)

            # Настройка главной карты
            ax_map.set_xlim(lon_bounds)
            ax_map.set_ylim(lat_bounds)
            ax_map.set_facecolor(self.colors["water"])

            # Устанавливаем границы в нужной проекции
            ax_map.set_global()
            ax_map.set_extent([lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]], crs=ccrs.PlateCarree())

            self._draw_earth_map(ax_map, lat_bounds, lon_bounds)

            # Информация о времени
            current_time = trajectory.iloc[frame]["analysis_time"]
            hours_elapsed = (current_time - start_time).total_seconds() / 3600
            current_day = int(hours_elapsed // 24) + 1

            # Траектория по дням
            if frame > 0:
                for i in range(frame + 1):
                    point_time = trajectory.iloc[i]["analysis_time"]
                    hours_from_start = (point_time - start_time).total_seconds() / 3600
                    day_color = self._get_day_color(hours_from_start)

                    lat = trajectory.iloc[i]["lat_deg"]
                    lon = trajectory.iloc[i]["lon_deg"]

                    size = 4 + 3 * (i / frame) if frame > 0 else 4
                    alpha = 0.4 + 0.3 * (i / frame) if frame > 0 else 0.7

                    ax_map.plot(
                        lon, lat, "o", color=day_color, markersize=size, alpha=alpha, transform=ccrs.PlateCarree()
                    )

                    # Сплошные линии для реальной траектории
                    if i < frame and i + 1 < len(trajectory):
                        next_lat = trajectory.iloc[i + 1]["lat_deg"]
                        next_lon = trajectory.iloc[i + 1]["lon_deg"]
                        ax_map.plot(
                            [lon, next_lon],
                            [lat, next_lat],
                            "-",
                            color=day_color,
                            linewidth=2,
                            alpha=1.0,
                            transform=ccrs.PlateCarree(),
                        )

            # Текущая позиция
            if frame >= len(trajectory):
                frame = len(trajectory) - 1

            current_lat = trajectory.iloc[frame]["lat_deg"]
            current_lon = trajectory.iloc[frame]["lon_deg"]
            current_pressure = trajectory.iloc[frame]["central_pressure_hpa"]
            current_grade = trajectory.iloc[frame]["grade"]

            pulse = 1 + 0.2 * np.sin(frame * 0.3)
            ax_map.plot(
                current_lon,
                current_lat,
                "o",
                color=self.colors["current"],
                markersize=12 * pulse,
                zorder=10,
                transform=ccrs.PlateCarree(),
            )

            # Прогнозы
            forecast_lat_text = "—"
            forecast_lon_text = "—"

            if frame in forecasts:
                forecast = forecasts[frame]
                all_forecasts.append(forecast)

                forecast_lat_text = f"{forecast['forecast_lat']:.1f}°N"
                forecast_lon_text = f"{forecast['forecast_lon']:.1f}°E"

                # Конус неопределенности
                p95_km = self.errors.get(horizon_hours, {}).get("p95_km", 200)
                radius_deg = p95_km / 111
                circle = Circle(
                    (forecast["forecast_lon"], forecast["forecast_lat"]),
                    radius_deg,
                    facecolor=self.colors["uncertainty"],
                    alpha=0.4,
                    edgecolor="gray",
                    linewidth=1,
                )
                ax_map.add_patch(circle)

                # Стрелка прогноза
                ax_map.annotate(
                    "",
                    xy=(forecast["forecast_lon"], forecast["forecast_lat"]),
                    xytext=(current_lon, current_lat),
                    arrowprops=dict(arrowstyle="->", color=self.colors["forecast"], lw=4, alpha=1.0),
                    xycoords=ccrs.PlateCarree()._as_mpl_transform(ax_map),
                    textcoords=ccrs.PlateCarree()._as_mpl_transform(ax_map),
                )

                # Прогнозная точка
                ax_map.plot(
                    forecast["forecast_lon"],
                    forecast["forecast_lat"],
                    "D",
                    color=self.colors["forecast"],
                    markersize=8,
                    zorder=8,
                    transform=ccrs.PlateCarree(),
                )

            # Пунктирные линии для прогнозной траектории
            if len(all_forecasts) > 1:
                for i in range(len(all_forecasts) - 1):
                    forecast_time = all_forecasts[i]["current_time"]
                    hours_from_start = (forecast_time - start_time).total_seconds() / 3600
                    day_color = self._get_day_color(hours_from_start)

                    ax_map.plot(
                        [all_forecasts[i]["forecast_lon"], all_forecasts[i + 1]["forecast_lon"]],
                        [all_forecasts[i]["forecast_lat"], all_forecasts[i + 1]["forecast_lat"]],
                        linestyle=(0, (3, 2)),
                        color=day_color,
                        linewidth=2,
                        alpha=1.0,
                        transform=ccrs.PlateCarree(),
                    )

                for i, forecast in enumerate(all_forecasts):
                    forecast_time = forecast["current_time"]
                    hours_from_start = (forecast_time - start_time).total_seconds() / 3600
                    day_color = self._get_day_color(hours_from_start)

                    ax_map.plot(
                        forecast["forecast_lon"],
                        forecast["forecast_lat"],
                        "D",
                        color=day_color,
                        markersize=5,
                        alpha=1.0,
                        zorder=7,
                        transform=ccrs.PlateCarree(),
                    )

            # МЕТАИНФОРМАЦИЯ (вверху)
            meta_text = f"{storm_name} ({cyclone_id}) | "
            meta_text += f"{current_time.strftime('%Y-%m-%d %H:%M UTC')} | "
            meta_text += f"Реальная: {current_lat:.1f}°N, {current_lon:.1f}°E | "
            meta_text += f"Прогноз (+{horizon_hours}ч): {forecast_lat_text}, {forecast_lon_text} | "
            meta_text += f"{current_pressure:.0f} hPa | Кат.{current_grade} | "
            meta_text += f"День {current_day} | Кадр {frame+1}/{len(trajectory)}"

            # Разбиваем длинный текст на строки для лучшего отображения
            if len(meta_text) > 120:
                parts = meta_text.split(" | ")
                line1 = " | ".join(parts[:3])
                line2 = " | ".join(parts[3:])
                meta_text = f"{line1}\n{line2}"

            ax_meta.text(
                0.5,
                0.5,
                meta_text,
                transform=ax_meta.transAxes,
                fontsize=13,
                verticalalignment="center",
                horizontalalignment="center",
                fontweight="bold",
                color="black",
            )

            # ЛЕГЕНДА (внизу)
            legend_elements = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=self.colors["current"],
                    markersize=8,
                    label="Текущая позиция",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="D",
                    color="w",
                    markerfacecolor=self.colors["forecast"],
                    markersize=7,
                    label=f"Прогноз +{horizon_hours}ч",
                ),
                plt.Line2D([0], [0], color=self.day_colors[0], linewidth=6, alpha=0.5, label="─ Реальная траектория"),
                plt.Line2D(
                    [0],
                    [0],
                    color=self.day_colors[0],
                    linewidth=6,
                    linestyle=(0, (3, 2)),
                    alpha=0.9,
                    label="- - Прогнозная траектория",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=self.colors["uncertainty"],
                    markersize=6,
                    alpha=0.6,
                    label=f'95% конус ({self.errors.get(horizon_hours, {}).get("p95_km", 200)} км)',
                ),
            ]

            ax_legend.legend(
                handles=legend_elements,
                loc="center",
                ncol=5,
                framealpha=0,
                fontsize=12,
                fancybox=False,
                shadow=False,
                borderpad=0,
                columnspacing=1.0,
                frameon=False,
            )

        # Создание анимации
        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=len(trajectory),
            interval=interval_ms,
            repeat=repeat,
            blit=False,
        )

        # Сохранение
        if output_path:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{cyclone_name.lower()}_{horizon_hours}h.mp4"
            logger.info(f"Сохранение анимации...")
            anim.save(str(output_file), writer="ffmpeg", fps=fps, bitrate=bitrate, dpi=dpi)
            logger.info(f"Анимация сохранена: {output_file}")

        plt.tight_layout(pad=0.5)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        if output_path is None:
            plt.show()
        else:
            plt.close(fig)

        return anim


def cli() -> None:
    parser = argparse.ArgumentParser(description="Анимации траекторий тропических циклонов")
    parser.add_argument("--data-path", type=str, default="bst_data/bst_all.csv", help="Путь к CSV с данными BST")
    parser.add_argument("--model-path", type=str, default="weights/model.onnx", help="Путь к модели ONNX")
    parser.add_argument(
        "--cyclone-ids", type=int, nargs="+", default=[cid for cid, _, _ in FIXED_CYCLONES], help="ID циклонов"
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=[12, 24, 48, 72], help="Горизонты прогноза, часы")
    parser.add_argument("--output-dir", type=str, default="cyclone_animations", help="Директория для вывода")
    parser.add_argument("--show", action="store_true", help="Показать вместо сохранения")
    parser.add_argument("--fps", type=float, default=2.5)
    parser.add_argument("--bitrate", type=int, default=1800)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--interval-ms", type=int, default=400)
    parser.add_argument("--repeat", action="store_true", default=True)
    parser.add_argument("--no-progress", action="store_true", help="Отключить прогресс-бар")
    parser.add_argument("--log-level", type=str, default="INFO", help="Уровень логирования: DEBUG/INFO/WARN/ERROR")
    parser.add_argument("--sample-every", type=int, default=2, help="Шаг отображения точек (2 => каждые 12ч)")
    parser.add_argument(
        "--skip-initial-points", type=int, default=1, help="Сколько первых точек пропускать для прогноза"
    )
    parser.add_argument("--margin-percent", type=float, default=0.15, help="Процентный отступ карты")
    parser.add_argument("--min-margin-deg", type=float, default=15, help="Минимальный отступ карты в градусах")

    args = parser.parse_args()

    level = getattr(logging, args.log_level.upper(), logging.INFO)
    animator = CycloneAnimator(
        data_path=args.data_path,
        model_path=args.model_path,
        show_progress=not args.no_progress,
        log_level=level,
    )

    output_dir = None if args.show else args.output_dir
    if output_dir and output_dir.strip():
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Создание анимаций для {len(args.cyclone_ids)} циклонов с горизонтами {args.horizons}")

    for cyclone_id in args.cyclone_ids:
        cyclone_name = next((name for cid, name, _ in FIXED_CYCLONES if cid == cyclone_id), str(cyclone_id))
        logger.info(f"\nОбработка {cyclone_name} ({cyclone_id})...")
        for horizon in args.horizons:
            try:
                logger.info(f"  Горизонт: {horizon} часов")
                animator.create_animation(
                    cyclone_id,
                    horizon,
                    output_path=output_dir,
                    sample_every=args.sample_every,
                    skip_initial_points=args.skip_initial_points,
                    margin_percent=args.margin_percent,
                    min_margin_deg=args.min_margin_deg,
                    interval_ms=args.interval_ms,
                    fps=args.fps,
                    bitrate=args.bitrate,
                    dpi=args.dpi,
                    repeat=args.repeat,
                )
            except Exception as e:  # noqa: BLE001
                logger.error(f"  Ошибка для горизонта {horizon}ч: {e}")
                continue

    if not output_dir:
        logger.info("Показ анимаций завершён")
    else:
        logger.info(f"\nВсе анимации сохранены в: {output_dir}")


if __name__ == "__main__":
    cli()
