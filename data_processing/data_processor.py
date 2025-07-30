import pandas as pd
from tqdm.auto import tqdm

from .base_processor import BaseDataProcessor
from .dataset_models import ProcessedDataset, SequenceConfig


class DataProcessor(BaseDataProcessor):
    """
    Процессор данных для прогнозирования траекторий циклонов.

    Attributes
    ----------
    feature_cfg : FeatureConfig
        Конфигурация фич с реестром
    horizons_hours : list[int]
        Список горизонтов прогноза в часах
    seq_config : SequenceConfig
        Конфигурация последовательностей
    validate_data : bool
        Флаг валидации данных
    train_max_year : int
        Максимальный год для обучения
    val_max_year : int
        Максимальный год для валидации
    """

    def __init__(
        self,
        horizons_hours: list[int],
        train_max_year: int,
        val_max_year: int,
        sequence_config: SequenceConfig | None = None,
        validate_data: bool = True,
    ):
        """
        Инициализирует процессор данных.

        Parameters
        ----------
        horizons_hours : list[int]
            Список горизонтов прогноза в часах
        train_max_year : int
            Максимальный год для обучающих данных
        val_max_year : int
            Максимальный год для валидационных данных
        sequence_config : SequenceConfig | None
            Конфигурация последовательностей
        validate_data : bool
            Выполнять ли валидацию данных
        """
        if not horizons_hours:
            raise ValueError("horizons_hours не может быть пустым")

        if val_max_year <= train_max_year:
            raise ValueError(f"val_max_year ({val_max_year}) должен быть больше train_max_year ({train_max_year})")

        super().__init__(sequence_config, validate_data)
        self.horizons_hours = horizons_hours
        self.train_max_year = train_max_year
        self.val_max_year = val_max_year

    def build_dataset(self, df: pd.DataFrame, horizon_hours: int) -> ProcessedDataset:
        """
        Строит обучающий датасет.

        Parameters
        ----------
        df : pd.DataFrame
            Входной датафрейм
        horizon_hours : int
            Горизонт прогноза в часах

        Returns
        -------
        ProcessedDataset
            Обработанный датасет
        """
        # Предобрабатываем данные
        df = self.preprocess_data(df)

        # Создаем примеры
        try:
            examples = self._create_examples(df)

            if not examples:
                raise ValueError("Не найдено валидных примеров для обучения")

            # Создаем финальный датафрейм
            df_final = pd.DataFrame(examples)
        except Exception as e:
            print(f"❌ Ошибка в _create_training_examples: {e}")
            import traceback

            traceback.print_exc()
            raise

        # Валидируем готовый датасет
        if self.validate_data:
            self.feature_cfg.validator.validate_sequences_format(df_final)
            self.feature_cfg.validator.validate_target_values(df_final)

        # Разделяем на X и y
        X = df_final.drop(columns=self.feature_cfg.target_features)
        y = df_final[self.feature_cfg.target_features]

        return ProcessedDataset(
            X=X, y=y, times=df_final["analysis_time"], intl_ids=df_final["intl_id"], storm_names=df_final["storm_name"]
        )

    def _create_examples(self, df: pd.DataFrame) -> list[dict]:
        """
        Создает обучающие примеры.

        Parameters
        ----------
        df : pd.DataFrame
            Предобработанный датафрейм

        Returns
        -------
        list[dict]
            Список примеров
        """
        examples = []

        for cid, cdf in tqdm(df.groupby("intl_id"), total=df["intl_id"].nunique(), desc="Creating training examples"):
            cdf = cdf.reset_index(drop=True)

            if len(cdf) < self.seq_config.min_history_length:
                continue

            for i in range(self.seq_config.min_history_length - 1, len(cdf)):
                curr = cdf.iloc[i]

                for horizon in self.horizons_hours:
                    example = self._create_single_example(cdf, i, curr, horizon)
                    if example:
                        examples.append(example)

        return examples

    def _create_single_example(self, cdf: pd.DataFrame, i: int, curr: pd.Series, horizon: int) -> dict | None:
        """
        Создает один обучающий пример.

        Parameters
        ----------
        cdf : pd.DataFrame
            Траектория циклона
        i : int
            Индекс текущей точки
        curr : pd.Series
            Текущая точка
        horizon : int
            Горизонт прогноза

        Returns
        -------
        dict | None
            Пример или None
        """
        # Получаем последовательность
        seq_df = self._get_sequence_slice(cdf, i)

        # Создаем последовательностные фичи
        sequences = self.feature_cfg.create_enhanced_sequence(seq_df)

        # Ищем целевую точку
        future_time = curr["analysis_time"] + pd.Timedelta(hours=horizon)
        future_row = cdf[cdf["analysis_time"] == future_time]

        if future_row.empty:
            return None

        future_row = future_row.iloc[0]

        # Вычисляем целевые значения
        dlat_target = future_row["lat_deg"] - curr["lat_deg"]
        dlon_target = future_row["lon_deg"] - curr["lon_deg"]

        # Вычисляем статические фичи
        static_features = self.feature_cfg.compute_static_features(curr, horizon)

        # Создаем пример
        example = {
            "sequences": sequences,
            "analysis_time": curr["analysis_time"],
            "intl_id": curr["intl_id"],
            "storm_name": curr.get("storm_name", "unknown"),
            "dlat_target": dlat_target,
            "dlon_target": dlon_target,
            **static_features,
        }

        return example
