import pandas as pd
from tqdm import tqdm

from coordinate_utils import CoordinateProcessor
from features import Feature, _inst, feature_classes


def preprocess_cyclone_data(df: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
    """
    Обрабатывает данные циклонов, добавляя признаки.

    Parameters
    ----------
    df : pd.DataFrame
        Исходные данные циклонов
    show_progress : bool
        Показывать прогресс-бар

    Returns
    -------
    pd.DataFrame
        Обработанные данные с признаками
    """
    df_sorted = df.sort_values(["intl_id", "analysis_time"]).reset_index(drop=True)
    Feature.reset_cyclone_cache()

    feats = [_inst(cls) for cls in feature_classes]
    columns: dict[str, list] = {cls.name(): [] for cls in feature_classes}

    iterator = df_sorted.itertuples(index=False, name=None)
    if show_progress:
        iterator = tqdm(iterator, total=len(df_sorted), desc="Calculating features")

    for row in iterator:
        row_s = pd.Series(row, index=df_sorted.columns)
        for f_obj, cls in zip(feats, feature_classes):
            columns[cls.name()].append(f_obj.calculate(row_s))

    out = df_sorted.copy()
    for col, values in columns.items():
        out[col] = values
    return out


def build_dataset(
    df: pd.DataFrame, horizons_hours: list[int] = [6], validate_data: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Создает датасет для обучения модели прогнозирования траекторий.

    Parameters
    ----------
    df : pd.DataFrame
        Обработанные данные циклонов
    horizons_hours : list[int]
        Список горизонтов прогноза в часах
    validate_data : bool
        Валидировать данные на корректность

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]
        X, y, times, intl_ids, storm_names
    """
    df = df.copy()
    df["analysis_time"] = pd.to_datetime(df["analysis_time"])

    # Нормализуем долготу в диапазон [-180, 180]
    df["lon_deg"] = CoordinateProcessor.normalize_longitude(df["lon_deg"])

    if validate_data:
        _validate_input_data(df)

    datasets = []

    for horizon in horizons_hours:
        df_h = df.copy()
        df_h["target_time"] = df_h["analysis_time"] + pd.Timedelta(hours=horizon)
        df_h["target_time_hours"] = horizon

        merged = pd.merge(
            df_h,
            df[["intl_id", "analysis_time", "lat_deg", "lon_deg"]],
            left_on=["intl_id", "target_time"],
            right_on=["intl_id", "analysis_time"],
            how="inner",
            suffixes=("", "_target"),
        )

        # Только относительные смещения
        merged["dlat_target"] = merged["lat_deg_target"] - merged["lat_deg"]
        merged["dlon_target"] = CoordinateProcessor.circ_diff(merged["lon_deg"], merged["lon_deg_target"])

        datasets.append(merged)

    df_final = pd.concat(datasets, ignore_index=True)

    # Валидация финального датасета
    if validate_data:
        _validate_final_dataset(df_final)

    feature_cols = ["target_time_hours"] + [f.name() for f in feature_classes]
    target_cols = ["dlat_target", "dlon_target"]  # Только относительные смещения

    X = df_final[feature_cols]
    y = df_final[target_cols]
    times = df_final["analysis_time"]
    intl_ids = df_final["intl_id"]
    storm_names = df_final["storm_name"] if "storm_name" in df_final.columns else pd.Series([""] * len(df_final))

    return X, y, times, intl_ids, storm_names


def split_data_by_years(
    X: pd.DataFrame,
    y: pd.DataFrame,
    times: pd.Series,
    train_max_year: int = 2017,
    val_max_year: int = 2019,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Разделяет данные по годам на train/val/test.

    Parameters
    ----------
    X, y, times, intl_ids, storm_names : данные
    train_max_year : int
        Максимальный год для обучающей выборки
    val_max_year : int
        Максимальный год для валидационной выборки

    Returns
    -------
    Tuple
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    mask_train = times.dt.year <= train_max_year
    mask_val = (times.dt.year > train_max_year) & (times.dt.year <= val_max_year)
    mask_test = times.dt.year > val_max_year

    X_train, y_train = X[mask_train], y[mask_train]
    X_val, y_val = X[mask_val], y[mask_val]
    X_test, y_test = X[mask_test], y[mask_test]

    return X_train, y_train, X_val, y_val, X_test, y_test


def print_dataset_info(
    X: pd.DataFrame,
    y: pd.DataFrame,
    times: pd.Series,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    train_mask: pd.Series,
    val_mask: pd.Series,
    test_mask: pd.Series,
) -> None:
    """
    Выводит информацию о датасете.
    """
    print("\n📊 ИТОГОВАЯ ИНФОРМАЦИЯ О ПАРАМЕТРИЗОВАННОМ ДАТАСЕТЕ:")
    print(f"  • Примеров для обучения: {len(X):,}")
    print(f"  • Признаков ({X.shape[1]}): {list(X.columns)}")
    print(f"  • Целевых переменных ({y.shape[1]}): {list(y.columns)}")

    print("\n📈 РАЗДЕЛЕНИЕ ДАННЫХ:")
    print(f"  • Train размер: {len(X_train):,}")
    print(f"  • Val   размер: {len(X_val):,}")
    print(f"  • Test  размер: {len(X_test):,}")
    print(f"  • Годы Train: {times[train_mask].dt.year.min()} — {times[train_mask].dt.year.max()}")
    print(f"  • Годы Val  : {times[val_mask].dt.year.min()} — {times[val_mask].dt.year.max()}")
    print(f"  • Годы Test : {times[test_mask].dt.year.min()} — {times[test_mask].dt.year.max()}")

    print("\n🎯 РАСПРЕДЕЛЕНИЕ ПО ГОРИЗОНТАМ ПРОГНОЗА:")
    if "target_time_hours" in X.columns:
        horizon_distribution = X["target_time_hours"].value_counts().sort_index()
        for horizon, count in horizon_distribution.items():
            percentage = count / len(X) * 100
            print(f"  • {horizon:2.0f}ч: {count:6,} примеров ({percentage:5.1f}%)")


def _validate_input_data(df: pd.DataFrame) -> None:
    """Валидирует входные данные."""
    required_cols = ["intl_id", "analysis_time", "lat_deg", "lon_deg"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing_cols}")

    if df["lat_deg"].isna().any():
        raise ValueError("Найдены NaN значения в lat_deg")

    if df["lon_deg"].isna().any():
        raise ValueError("Найдены NaN значения в lon_deg")

    if (df["lat_deg"].abs() > 90).any():
        raise ValueError("Некорректные значения широты (вне [-90, 90])")

    if (df["lon_deg"].abs() > 180).any():
        raise ValueError("Некорректные значения долготы (вне [-180, 180])")


def _validate_final_dataset(df: pd.DataFrame) -> None:
    """Валидирует финальный датасет."""
    required_target_cols = ["dlat_target", "dlon_target"]
    missing_cols = [col for col in required_target_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Отсутствуют целевые колонки: {missing_cols}")

    # Проверяем на экстремальные значения смещений
    dlat_extreme = df["dlat_target"].abs() > 10  # более 10 градусов за один шаг
    dlon_extreme = df["dlon_target"].abs() > 20  # более 20 градусов за один шаг

    if dlat_extreme.any():
        n_extreme = dlat_extreme.sum()
        print(f"⚠️ Предупреждение: найдено {n_extreme} экстремальных смещений по широте (>10°)")

    if dlon_extreme.any():
        n_extreme = dlon_extreme.sum()
        print(f"⚠️ Предупреждение: найдено {n_extreme} экстремальных смещений по долготе (>20°)")


class DataProcessor:
    """
    Класс для полной обработки данных циклонов.
    """

    def __init__(
        self,
        horizons_hours: list[int] = [6, 12, 18, 24, 48],
        train_max_year: int = 2017,
        val_max_year: int = 2019,
        validate_data: bool = True,
    ):
        """
        Parameters
        ----------
        horizons_hours : list[int]
            Горизонты прогноза в часах
        train_max_year : int
            Максимальный год для обучающей выборки
        val_max_year : int
            Максимальный год для валидационной выборки
        validate_data : bool
            Валидировать данные
        """
        self.horizons_hours = horizons_hours
        self.train_max_year = train_max_year
        self.val_max_year = val_max_year
        self.validate_data = validate_data

    def process_full_pipeline(self, df: pd.DataFrame) -> dict:
        """
        Выполняет полный пайплайн обработки данных.

        Parameters
        ----------
        df : pd.DataFrame
            Исходные данные

        Returns
        -------
        dict
            Словарь с обработанными данными
        """
        print("🔄 Начинаем обработку данных...")

        # 1. Препроцессинг признаков
        print("📊 Вычисляем признаки...")
        df_processed = preprocess_cyclone_data(df)

        # 2. Создание датасета
        print("🎯 Создаем датасет...")
        X, y, times, intl_ids, storm_names = build_dataset(
            df_processed,
            horizons_hours=self.horizons_hours,
            validate_data=self.validate_data,
        )

        # 3. Разделение данных
        print("✂️ Разделяем данные...")
        X_train, y_train, X_val, y_val, X_test, y_test = split_data_by_years(
            X, y, times, train_max_year=self.train_max_year, val_max_year=self.val_max_year
        )

        # 4. Создание масок для совместимости
        mask_train = times.dt.year <= self.train_max_year
        mask_val = (times.dt.year > self.train_max_year) & (times.dt.year <= self.val_max_year)
        mask_test = times.dt.year > self.val_max_year

        # 5. Вывод информации
        print_dataset_info(X, y, times, X_train, X_val, X_test, mask_train, mask_val, mask_test)

        return {
            "df_processed": df_processed,
            "X": X,
            "y": y,
            "times": times,
            "intl_ids": intl_ids,
            "storm_names": storm_names,
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "mask_train": mask_train,
            "mask_val": mask_val,
            "mask_test": mask_test,
        }
