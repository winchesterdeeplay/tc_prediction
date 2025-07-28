from typing import cast

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


class CatBoostLatLon:
    """
    CatBoost-модель для предсказания смещений координат (dlat, dlon).
    """

    def __init__(self, **params):
        """
        Parameters
        ----------
        **params : параметры для CatBoostRegressor
        """
        default_params = {
            "loss_function": "MultiRMSE",
            "iterations": 2000,
            "learning_rate": 0.05,
            "depth": 6,
            "early_stopping_rounds": 200,
            "use_best_model": True,
            "random_seed": 42,
            "verbose": 0,
        }
        default_params.update(params)
        self.model = CatBoostRegressor(**default_params)
        self.feature_names_ = None
        self.is_fitted_ = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_val: pd.DataFrame | None = None,
        y_val: pd.DataFrame | None = None,
        sample_weight: np.ndarray | None = None,
        sample_weight_val: np.ndarray | None = None,
        cat_features: list[str] | None = None,
    ):
        """
        Обучает модель.

        Parameters
        ----------
        X : pd.DataFrame
            Обучающие признаки
        y : pd.DataFrame
            Обучающие таргеты (должен содержать dlat_target, dlon_target)
        X_val : pd.DataFrame, optional
            Валидационные признаки
        y_val : pd.DataFrame, optional
            Валидационные таргеты
        sample_weight : np.ndarray, optional
            Веса для обучающих примеров
        sample_weight_val : np.ndarray, optional
            Веса для валидационных примеров
        cat_features : List[str], optional
            Категориальные признаки
        """
        # Проверяем наличие необходимых колонок
        required_cols = ["dlat_target", "dlon_target"]
        missing_cols = [col for col in required_cols if col not in y.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют обязательные колонки в y: {missing_cols}")

        # Подготавливаем данные
        y_train = y[required_cols]

        eval_set: list[tuple[pd.DataFrame, pd.DataFrame, np.ndarray | None]] | None = None
        if X_val is not None and y_val is not None:
            y_val_subset = y_val[required_cols]
            if sample_weight_val is not None:
                eval_set = [(X_val, y_val_subset, sample_weight_val)]
            else:
                eval_set = [cast(tuple[pd.DataFrame, pd.DataFrame, np.ndarray | None], (X_val, y_val_subset))]

        # Определяем категориальные признаки
        if cat_features is None:
            cat_features = ["target_time_hours", "grade"] if "grade" in X.columns else ["target_time_hours"]

        # Обучаем модель
        self.model.fit(X, y_train, eval_set=eval_set, sample_weight=sample_weight, cat_features=cat_features)

        self.feature_names_ = list(X.columns)
        self.is_fitted_ = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Делает предсказания.

        Parameters
        ----------
        X : pd.DataFrame
            Признаки для предсказания

        Returns
        -------
        np.ndarray
            Предсказания формы (n_samples, 2) для (dlat, dlon)
        """
        if not self.is_fitted_:
            raise ValueError("Модель не обучена. Вызовите fit() перед predict()")

        return self.model.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Возвращает важности признаков."""
        if not self.is_fitted_:
            raise ValueError("Модель не обучена")
        return self.model.feature_importances_

    @property
    def feature_names(self) -> list[str]:
        """Возвращает названия признаков."""
        if not self.is_fitted_:
            raise ValueError("Модель не обучена")
        return self.feature_names_


def calculate_sample_weights(X: pd.DataFrame, strategy: str = "long_horizon_focus") -> np.ndarray:
    """
    Вычисляет веса для сэмплов в зависимости от стратегии.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame с признаками (должен содержать target_time_hours)
    strategy : str
        Стратегия взвешивания

    Returns
    -------
    np.ndarray
        Веса для каждого примера
    """
    if "target_time_hours" not in X.columns:
        return np.ones(len(X))

    if strategy == "balanced_horizon":
        # Балансировка между горизонтами
        horizon_counts = X["target_time_hours"].value_counts()
        max_count = horizon_counts.max()
        weights = X["target_time_hours"].map(lambda h: max_count / horizon_counts[h])

    elif strategy == "inverse_horizon":
        # Больший вес для коротких горизонтов
        max_horizon = X["target_time_hours"].max()
        weights = X["target_time_hours"].map(lambda h: max_horizon / h)

    elif strategy == "long_horizon_focus":
        # Больший вес для дальних горизонтов (линейная зависимость)
        min_horizon = X["target_time_hours"].min()
        weights = X["target_time_hours"].map(lambda h: h / min_horizon)

    elif strategy == "long_horizon_quadratic":
        # Больший вес для дальних горизонтов (квадратичная зависимость)
        min_horizon = X["target_time_hours"].min()
        weights = X["target_time_hours"].map(lambda h: (h / min_horizon) ** 1.5)

    elif strategy == "distance_based":
        # Вес на основе скорости (более надежные предсказания для медленных циклонов)
        if "speed_prev" in X.columns:
            speeds = X["speed_prev"].fillna(X["speed_prev"].median())
            max_speed = speeds.quantile(0.95)  # избегаем выбросов
            weights = (max_speed - speeds.clip(0, max_speed)) / max_speed + 0.1
        else:
            weights = pd.Series(1.0, index=X.index)

    else:
        weights = pd.Series(1.0, index=X.index)

    # Нормализация весов
    weights = weights / weights.mean()
    return weights.values


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    use_weights: bool = True,
    weight_strategy: str = "long_horizon_focus",
    model_params: dict | None = None,
    verbose: bool = True,
) -> CatBoostLatLon:
    """
    Обучает модель CatBoost.

    Parameters
    ----------
    X_train, y_train : обучающие данные
    X_val, y_val : валидационные данные
    use_weights : bool
        Использовать ли веса для сэмплов
    weight_strategy : str
        Стратегия взвешивания
    model_params : Dict, optional
        Дополнительные параметры модели
    verbose : bool
        Выводить информацию об обучении

    Returns
    -------
    CatBoostLatLon
        Обученная модель
    """
    if verbose:
        print(f"📊 Обучающих примеров: {len(X_train):,}")
        print(f"📊 Признаков: {X_train.shape[1]}")

        if "target_time_hours" in X_train.columns:
            horizon_counts = X_train["target_time_hours"].value_counts().sort_index()
            print("📈 Примеры по горизонтам:")
            for h, count in horizon_counts.items():
                print(f"    {h:2.0f}ч: {count:5,} примеров")

    # Вычисляем веса для сэмплов
    sample_weight = None
    if use_weights:
        sample_weight = calculate_sample_weights(X_train, strategy=weight_strategy)

        if verbose:
            print(f"⚖️ Используются веса ({weight_strategy}):")
            print(f"   Мин. вес: {sample_weight.min():.3f}, Макс. вес: {sample_weight.max():.3f}")
            print(f"   Средний вес: {sample_weight.mean():.3f}")

    # Создаем модель
    params = model_params or {}
    model = CatBoostLatLon(**params)

    # Обучаем модель
    model.fit(X_train, y_train, X_val, y_val, sample_weight=sample_weight)

    if verbose:
        weight_info = f" с весами ({weight_strategy})" if use_weights else " без весов"
        print(f"✅ CatBoost обучен{weight_info} (мультирегрессия dlat и dlon)")

    return model


class ModelTrainer:
    """
    Класс для управления процессом обучения модели.
    """

    def __init__(
        self,
        model_params: dict | None = None,
        use_weights: bool = True,
        weight_strategy: str = "balanced_horizon",
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        model_params : Dict, optional
            Параметры для CatBoost модели
        use_weights : bool
            Использовать веса для сэмплов
        weight_strategy : str
            Стратегия взвешивания
        verbose : bool
            Выводить прогресс
        """
        self.model_params = model_params or {}
        self.use_weights = use_weights
        self.weight_strategy = weight_strategy
        self.verbose = verbose
        self.model: CatBoostLatLon | None = None

    def train(
        self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame
    ) -> CatBoostLatLon:
        """
        Обучает модель.

        Returns
        -------
        CatBoostLatLon
            Обученная модель
        """
        self.model = train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            use_weights=self.use_weights,
            weight_strategy=self.weight_strategy,
            model_params=self.model_params,
            verbose=self.verbose,
        )
        return self.model

    def get_feature_importance_analysis(self) -> dict:
        """
        Анализирует важность признаков.

        Returns
        -------
        Dict
            Результаты анализа важности признаков
        """
        if self.model is None:
            raise ValueError("Модель не обучена")

        feature_importance = self.model.feature_importances_
        feature_names = self.model.feature_names

        # Создаем DataFrame для анализа
        importance_df = pd.DataFrame({"feature": feature_names, "importance": feature_importance}).sort_values(
            "importance", ascending=False
        )

        return {
            "importance_df": importance_df,
            "total_features": len(feature_names),
            "top_10_features": importance_df.head(10)["feature"].tolist(),
            "importance_stats": {
                "mean": feature_importance.mean(),
                "std": feature_importance.std(),
                "max": feature_importance.max(),
                "min": feature_importance.min(),
            },
        }
