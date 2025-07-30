from typing import Any, Self

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import Module
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from core.coordinates import CoordinateProcessor
from core.features import FeatureConfig

from .losses import (
    CombinedCycloneLoss,
    HaversineLoss,
    NLLGaussianLoss,
    SectorLoss,
)


class SimpleTransformerModel(Module):
    def __init__(
        self,
        sequence_feature_dim: int,
        static_feature_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        output_dim: int = 2,
        max_seq_length: int = 1000,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.static_feature_dim = static_feature_dim
        self.sequence_feature_dim = sequence_feature_dim
        self.max_seq_length = max_seq_length

        # Проекция входных фич в hidden_dim
        self.input_projection = nn.Linear(sequence_feature_dim, hidden_dim)
        
        # Learnable Positional Encoding
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_length, hidden_dim))
        
        # Sinusoidal Positional Encoding
        # self.register_buffer('pos_encoding', self._create_sinusoidal_encoding(max_seq_length, hidden_dim))
        
        # Transformer Encoder слои
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        
        self.dropout = nn.Dropout(dropout)

        # Голова для статических фич
        self.static_head = nn.Sequential(
            nn.Linear(static_feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Объединяющий слой
        combined_dim = hidden_dim + hidden_dim // 2
        self.combined_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Выходной слой
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """
        Создает sinusoidal positional encoding.
        
        Args:
            max_len: Максимальная длина последовательности
            d_model: Размерность модели
            
        Returns:
            Positional encoding tensor [max_len, d_model]
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe

    def forward(
        self, sequences: torch.Tensor, static_features: torch.Tensor, seq_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Прямой проход через Transformer модель.

        Parameters:
        ----------
        sequences : Tensor
            Последовательности признаков [batch_size, seq_len, sequence_feature_dim]
        static_features : Tensor
            Статические признаки [batch_size, static_feature_dim]
        seq_lengths : Tensor
            Длины последовательностей [batch_size]

        Returns:
        ----------
        Tensor
            Предсказанные изменения координат (dlat, dlon)
        """
        batch_size, seq_len, feature_dim = sequences.shape
        
        # Проекция входных фич
        x = self.input_projection(sequences)  # [batch_size, seq_len, hidden_dim]
        
        # Добавляем positional encoding
        pos_enc = self.pos_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_enc
        
        # Создаем маску для padding (True для padding токенов)
        mask = torch.arange(seq_len, device=sequences.device).unsqueeze(0) >= seq_lengths.unsqueeze(1)
        
        # Transformer Encoder проход
        transformer_out = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Извлекаем последние состояния для каждой последовательности
        indices = torch.arange(batch_size, device=transformer_out.device)
        sequence_features = transformer_out[indices, seq_lengths - 1]
        
        # Dropout для последовательностных фич
        sequence_features = self.dropout(sequence_features)

        # Обработка статических фич
        static_features = self.static_head(static_features)

        # Объединяем фичи
        combined_features = torch.cat([sequence_features, static_features], dim=1)

        # Объединяющий слой
        combined_out = self.combined_head(combined_features)

        # Выходной слой
        return self.output_layer(combined_out)  # type: ignore[no-any-return]





class NNLatLon(Module):
    """
    Neural Network model for predicting cyclone trajectory changes.

    Inherits from SimpleGRUModel but adds prediction capabilities.
    """

    def __init__(
        self, sequence_feature_dim: int, static_feature_dim: int = 5, hidden_dim: int = 128, num_layers: int = 2, num_heads: int = 8, output_dim: int = 2
    ):
        super().__init__()
        self.model = SimpleTransformerModel(sequence_feature_dim, static_feature_dim, hidden_dim, num_layers, num_heads, output_dim=output_dim)

    def forward(
        self, sequences: torch.Tensor, static_features: torch.Tensor, seq_lengths: torch.Tensor
    ) -> torch.Tensor:
        result: torch.Tensor = self.model(sequences, static_features, seq_lengths)
        return result

    def predict(self, x: pd.DataFrame, batch_size: int = 256) -> np.ndarray:
        """Predict on a dataframe.

        Parameters
        ----------
        x : pd.DataFrame
            DataFrame that contains a column with sequences and static features.
        batch_size : int
            Batch size for prediction.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        feature_cfg = FeatureConfig()
        if feature_cfg.sequences_column not in x.columns:
            raise ValueError(f"DataFrame must contain '{feature_cfg.sequences_column}' column")

        return self._predict_dataframe(x, batch_size)

    def _predict_dataframe(self, x: pd.DataFrame, batch_size: int = 256) -> np.ndarray:
        """Predict from DataFrame with automatic padding."""
        if "sequences" not in x.columns:
            raise ValueError("DataFrame must contain 'sequences' column")

        device = next(self.parameters()).device

        # Извлекаем последовательности и статические фичи
        feature_cfg = FeatureConfig()
        sequences = [torch.tensor(seq, dtype=torch.float32) for seq in x[feature_cfg.sequences_column].tolist()]

        # Статические фичи (если есть)
        static_cols = feature_cfg.static_features
        if all(col in x.columns for col in static_cols):
            static_features = torch.tensor(x[static_cols].values, dtype=torch.float32)
        else:
            raise ValueError(f"Отсутствуют необходимые статические фичи: {static_cols}")

        preds: list[torch.Tensor] = []
        with torch.no_grad():
            # Обрабатываем батчами
            for i in range(0, len(sequences), batch_size):
                batch_sequences = sequences[i : i + batch_size]
                batch_static = static_features[i : i + batch_size]

                # Паддинг для батча
                max_len = max(seq.size(0) for seq in batch_sequences)
                batch_size_actual = len(batch_sequences)
                feature_dim = batch_sequences[0].size(1)

                padded_batch = torch.zeros((batch_size_actual, max_len, feature_dim))
                seq_lengths = torch.zeros(batch_size_actual, dtype=torch.long)

                for j, seq in enumerate(batch_sequences):
                    seq_len = seq.size(0)
                    padded_batch[j, :seq_len] = seq
                    seq_lengths[j] = seq_len

                padded_batch = padded_batch.to(device)
                batch_static = batch_static.to(device)
                seq_lengths = seq_lengths.to(device)

                out = self.forward(padded_batch, batch_static, seq_lengths)
                preds.append(out.cpu())

        return torch.cat(preds).numpy()

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame | None = None,
        y_val: pd.DataFrame | None = None,
        *,
        sample_weight: np.ndarray | None = None,
        criterion: nn.Module | None = None,
        batch_size: int = 64,
        epochs: int = 20,
        learning_rate: float = 1e-3,
        verbose: bool = True,
        save_on_interrupt: bool = False,
        interrupt_save_path: str = "nn_lat_lon.pt",
        shuffle_dataset: bool = False,
        **unused: Any,
    ) -> Self:
        """Simple supervised training loop.

        Parameters
        ----------
        X_train, y_train : pd.DataFrame
            Training feature/target data. ``X_train`` must contain sequences and static features.
        X_val, y_val : pd.DataFrame, optional
            Validation split. If ``None``, no validation will be performed.
        sample_weight : np.ndarray, optional
            Per-sample weights from the trainer.
        criterion : torch.nn.Module, optional
            Loss to optimise. Defaults to mean-squared error.
        batch_size, epochs, learning_rate : int | float
            Basic training hyper-parameters.
        verbose : bool
            If *True*, prints epoch progress.
        save_on_interrupt : bool
            If *True*, tries to save current parameters on ``KeyboardInterrupt``.
        interrupt_save_path : str
            Path where parameters will be stored.
        unused : dict
            Placeholder to absorb extra kwargs coming from ``ModelTrainer``.
        """
        from torch.utils.data import DataLoader

        from models.dataset import CycloneDataset, collate_variable_length

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # ------------------------------------------------------------------
        # Dataset / loader
        # ------------------------------------------------------------------
        # Создаем статические фичи, включая горизонты прогноза
        feature_cfg = FeatureConfig()
        static_cols = feature_cfg.static_features
        static_features_train = np.zeros((len(X_train), len(static_cols)), dtype=np.float32)
        for i, col in enumerate(static_cols):
            if col in X_train.columns:
                static_features_train[:, i] = X_train[col].values

        # Извлекаем горизонты отдельно для loss функций
        horizon_hours_train = X_train[feature_cfg.target_time_column].values.astype(np.float32)

        train_ds = CycloneDataset(
            sequences=X_train[feature_cfg.sequences_column].tolist(),
            static_features=static_features_train,
            y=y_train.values,
            horizon_hours=horizon_hours_train,
            sample_weight=sample_weight,
            shuffle_dataset=shuffle_dataset,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_variable_length,
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            static_features_val = np.zeros((len(X_val), len(static_cols)), dtype=np.float32)
            for i, col in enumerate(static_cols):
                if col in X_val.columns:
                    static_features_val[:, i] = X_val[col].values

            horizon_hours_val = X_val[feature_cfg.target_time_column].values.astype(np.float32)

            val_ds = CycloneDataset(
                sequences=X_val[feature_cfg.sequences_column].tolist(),
                static_features=static_features_val,
                y=y_val.values,
                horizon_hours=horizon_hours_val,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_variable_length,
            )

        # ------------------------------------------------------------------
        # Optimiser / loss
        # ------------------------------------------------------------------
        if criterion is None:
            criterion = nn.MSELoss(reduction="none" if sample_weight is not None else "mean")
        optimiser = torch.optim.Adam(self.parameters(), lr=learning_rate)

        try:
            for epoch in range(1, epochs + 1):
                self.train()
                epoch_loss = 0.0
                batch_iter = train_loader
                if verbose:
                    from tqdm.auto import tqdm

                    batch_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

                for batch in batch_iter:
                    padded_seqs, static_features, targets, weights, horizon_hours, seq_lengths = batch
                    padded_seqs = padded_seqs.to(device)
                    static_features = static_features.to(device)
                    targets = targets.to(device)
                    seq_lengths = seq_lengths.to(device)
                    weights = weights.to(device)

                    optimiser.zero_grad()
                    preds = self(padded_seqs, static_features, seq_lengths)

                    loss_per_sample = criterion(preds, targets)
                    if loss_per_sample.dim() > 1:
                        loss_per_sample = loss_per_sample.mean(dim=1)

                    if sample_weight is not None:
                        loss_tensor = (loss_per_sample * weights).mean()
                    else:
                        loss_tensor = loss_per_sample.mean()
                    loss_tensor.backward()
                    optimiser.step()
                    epoch_loss += loss_tensor.item()

                epoch_loss /= len(train_loader)
                if verbose:
                    msg = f"Epoch {epoch:>3}/{epochs} | train_loss={epoch_loss:.4f}"
                    if val_loader is not None:
                        val_loss = _evaluate(self, val_loader, criterion, device)
                        msg += f" | val_loss={val_loss:.4f}"
                    from tqdm.auto import tqdm

                    tqdm.write(msg)

        except KeyboardInterrupt:
            if save_on_interrupt:
                print(f"\n⚠️ Training interrupted, saving checkpoint to {interrupt_save_path}")
                self.save(interrupt_save_path)
            raise

        return self

    def save(self, path: str) -> None:
        """Save model parameters to ``path`` using :pymeth:`torch.save`."""
        torch.save(self.state_dict(), path)


def _evaluate(model: "NNLatLon", loader: Any, criterion: Any, device: torch.device) -> float:
    """Utility evaluation on a *loader* returning average loss."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            padded_seqs, static_features, targets, weights, horizon_hours, seq_lengths = batch

            padded_seqs = padded_seqs.to(device)
            static_features = static_features.to(device)
            targets = targets.to(device)
            seq_lengths = seq_lengths.to(device)
            preds = model(padded_seqs, static_features, seq_lengths)
            loss_tensor = criterion(preds, targets).mean()
            total += loss_tensor.item()
    return total / len(loader)


class LightningCycloneModel(pl.LightningModule):
    """LightningModule that wraps NNLatLon and handles optimisation/logging."""

    def __init__(
        self,
        sequence_feature_dim: int,
        static_feature_dim: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        learning_rate: float = 1e-3,
        loss_fn: str | nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Resolve loss function
        if loss_fn is None:
            # По умолчанию используем улучшенную horizon-aware loss для лучшего обучения
            self.criterion: nn.Module = SectorLoss()
        elif isinstance(loss_fn, str):
            loss_name = loss_fn.lower()
            if loss_name in {"mse", "mseloss"}:
                self.criterion = nn.MSELoss()
            elif loss_name in {"sector", "sectorloss"}:
                self.criterion = SectorLoss()
            elif loss_name in {"nll", "gaussian", "negloglik", "nllgaussian"}:
                self.criterion = NLLGaussianLoss()
            elif loss_name in {"haversine", "haversineloss"}:
                self.criterion = HaversineLoss()
            elif loss_name in {"combined", "combinedcyclone", "combinedcycloneloss"}:
                self.criterion = CombinedCycloneLoss()
            else:
                raise ValueError(f"Unknown loss function: {loss_fn}")
        else:
            self.criterion = loss_fn

        output_dim = 4 if isinstance(self.criterion, (NLLGaussianLoss)) else 2
        self.net = NNLatLon(
            sequence_feature_dim=sequence_feature_dim,
            static_feature_dim=static_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            output_dim=output_dim,
        )
        self.learning_rate = learning_rate
        self.feature_cfg = FeatureConfig()

        self._criterion_handles_weights = isinstance(
            self.criterion,
            (
                SectorLoss,
                NLLGaussianLoss,
                HaversineLoss,
                CombinedCycloneLoss,
            ),
        )

    def forward(
        self, sequences: torch.Tensor, static_features: torch.Tensor, seq_lengths: torch.Tensor
    ) -> torch.Tensor:
        return self.net(sequences, static_features, seq_lengths)  # type: ignore[no-any-return]

    def export_to_onnx(
        self,
        filepath: str,
        sequence_length: int = 50,
        batch_size: int = 1,
        dynamic_axes: bool = True,
        opset_version: int = 13,
        validate: bool = True,
        device: str = "cpu",
    ) -> None:
        """
        Экспортирует модель в ONNX формат.

        Parameters
        ----------
        filepath : str
            Путь для сохранения ONNX файла
        sequence_length : int
            Максимальная длина последовательности для экспорта
        batch_size : int
            Размер батча для экспорта (может быть динамическим)
        dynamic_axes : bool
            Использовать ли динамические оси для batch_size и sequence_length
        opset_version : int
            Версия ONNX операторов
        validate : bool
            Выполнять ли валидацию после экспорта
        """
        print(f"🔄 Создаем тестовые данные для экспорта модели в ONNX...")
        self.eval()

        feature_cfg = FeatureConfig()
        feature_dims = feature_cfg.get_feature_dimensions()

        # Создаем тестовые данные
        dummy_sequences = torch.randn(batch_size, sequence_length, feature_dims["sequence"]).to(device)
        dummy_static = torch.randn(batch_size, feature_dims["static"]).to(device)
        dummy_lengths = torch.randint(1, sequence_length + 1, (batch_size,)).to(device)

        # Настраиваем динамические оси
        if dynamic_axes:
            dynamic_axes_config = {
                "sequences": {0: "batch_size", 1: "sequence_length"},
                "static_features": {0: "batch_size"},
                "seq_lengths": {0: "batch_size"},
                "output": {0: "batch_size"},
            }
        else:
            dynamic_axes_config = None
            
        print("🔄 Начинаем экспорт модели в ONNX...")
        
        torch.onnx.export(
            self.net,
            (dummy_sequences, dummy_static, dummy_lengths),
            filepath,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["sequences", "static_features", "seq_lengths"],
            output_names=["output"],
            dynamic_axes=dynamic_axes_config,
            verbose=True,
            keep_initializers_as_inputs=False,
            export_modules_as_functions=False,
            dynamo=True
        )

        print(f"✅ Модель успешно экспортирована в ONNX: {filepath}")

        if validate:
            print("🔍 Валидируем экспортированную модель...")
            self._validate_onnx_model(filepath, dummy_sequences, dummy_static, dummy_lengths)

    def _validate_onnx_model(
        self, filepath: str, test_sequences: torch.Tensor, test_static: torch.Tensor, test_lengths: torch.Tensor
    ) -> None:
        """
        Валидирует экспортированную ONNX модель.

        Parameters
        ----------
        filepath : str
            Путь к ONNX файлу
        test_sequences : torch.Tensor
            Тестовые последовательности
        test_static : torch.Tensor
            Тестовые статические фичи
        test_lengths : torch.Tensor
            Тестовые длины последовательностей
        """
        try:
            # Загружаем ONNX модель
            onnx_model = onnx.load(filepath)
            onnx.checker.check_model(onnx_model)

            # Создаем ONNX Runtime сессию
            ort_session = ort.InferenceSession(filepath, providers=["CPUExecutionProvider"])

            # Получаем предсказания PyTorch модели
            with torch.no_grad():
                pytorch_output = self.net(test_sequences, test_static, test_lengths).numpy()

            # Получаем предсказания ONNX модели
            onnx_inputs = {
                "sequences": test_sequences.numpy(),
                "static_features": test_static.numpy(),
                "seq_lengths": test_lengths.numpy(),
            }
            onnx_output = ort_session.run(None, onnx_inputs)[0]

            # Сравниваем результаты
            diff = np.abs(pytorch_output - onnx_output).max()
            if diff < 1e-5:
                print(f"✅ ONNX модель валидирована успешно. Максимальная разница: {diff:.2e}")
            else:
                print(f"⚠️ Внимание: Разница между PyTorch и ONNX предсказаниями: {diff:.2e}")

        except Exception as e:
            print(f"❌ Ошибка валидации ONNX модели: {e}")

    @staticmethod
    def load_onnx_model(filepath: str) -> ort.InferenceSession:
        """
        Загружает ONNX модель для инференса.

        Parameters
        ----------
        filepath : str
            Путь к ONNX файлу

        Returns
        -------
        ort.InferenceSession
            ONNX Runtime сессия для инференса
        """
        return ort.InferenceSession(filepath, providers=["CPUExecutionProvider"])

    @staticmethod
    def predict_with_onnx(
        ort_session: ort.InferenceSession, sequences: np.ndarray, static_features: np.ndarray, seq_lengths: np.ndarray
    ) -> np.ndarray:
        """
        Делает предсказания используя ONNX модель.

        Parameters
        ----------
        ort_session : ort.InferenceSession
            ONNX Runtime сессия
        sequences : np.ndarray
            Последовательности фич [batch_size, seq_len, feature_dim]
        static_features : np.ndarray
            Статические фичи [batch_size, static_feature_dim]
        seq_lengths : np.ndarray
            Длины последовательностей [batch_size]

        Returns
        -------
        np.ndarray
            Предсказания [batch_size, output_dim]
        """
        try:
            inputs = {
                "sequences": sequences.astype(np.float32),
                "static_features": static_features.astype(np.float32),
                "seq_lengths": seq_lengths.astype(np.int64),
            }

            outputs = ort_session.run(None, inputs)
            return outputs[0]  # type: ignore[no-any-return]
        except Exception as e:
            print(f"❌ Ошибка при выполнении ONNX предсказания: {e}")
            print(
                f"   Размеры входов: sequences={sequences.shape}, static_features={static_features.shape}, seq_lengths={seq_lengths.shape}"
            )
            raise

    @staticmethod
    def _validate_input(X: pd.DataFrame, feature_cfg: FeatureConfig) -> None:
        feature_cfg.validator.validate_sequences_format(X)

        static_features = feature_cfg.static_features
        missing_static = [col for col in static_features if col not in X.columns]
        if missing_static:
            raise ValueError(f"Отсутствуют необходимые статические фичи: {missing_static}")

        if feature_cfg.sequences_column not in X.columns:
            raise ValueError(f"DataFrame must contain '{feature_cfg.sequences_column}' column")

        first_sequence = X[feature_cfg.sequences_column].iloc[0]
        if not isinstance(first_sequence, (list, np.ndarray)):
            raise ValueError(f"'{feature_cfg.sequences_column}' column must contain sequences (lists or arrays)")

    def predict(
        self,
        x: pd.DataFrame,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Предсказание на датафрейме.

        Parameters
        ----------
        x : pd.DataFrame
            Датафрейм с последовательностями и статическими фичами
        batch_size : int
            Размер батча для предсказания

        Returns
        -------
        np.ndarray
            Предсказания
        """
        self._validate_input(x, self.feature_cfg)

        device = next(self.parameters()).device

        # Извлекаем последовательности и статические фичи
        sequences = [torch.tensor(seq, dtype=torch.float32) for seq in x[self.feature_cfg.sequences_column].tolist()]
        static_cols = self.feature_cfg.static_features
        if all(col in x.columns for col in static_cols):
            static_features = torch.tensor(x[static_cols].values, dtype=torch.float32)
        else:
            raise ValueError(f"Отсутствуют необходимые статические фичи: {static_cols}")

        preds: list[torch.Tensor] = []
        with torch.no_grad():
            # Обрабатываем батчами
            for i in range(0, len(sequences), batch_size):
                batch_sequences = sequences[i : i + batch_size]
                batch_static = static_features[i : i + batch_size]

                # Паддинг для батча
                max_len = max(seq.size(0) for seq in batch_sequences)
                batch_size_actual = len(batch_sequences)
                feature_dim = batch_sequences[0].size(1)

                padded_batch = torch.zeros((batch_size_actual, max_len, feature_dim))
                seq_lengths = torch.zeros(batch_size_actual, dtype=torch.long)

                for j, seq in enumerate(batch_sequences):
                    seq_len = seq.size(0)
                    padded_batch[j, :seq_len] = seq
                    seq_lengths[j] = seq_len

                padded_batch = padded_batch.to(device)
                batch_static = batch_static.to(device)
                seq_lengths = seq_lengths.to(device)

                out = self.forward(padded_batch, batch_static, seq_lengths)
                preds.append(out.cpu())

        return torch.cat(preds).numpy()

    def _shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        """Общий шаг для train и validation."""
        padded_seqs, static_features, targets, weights, horizon_hours, seq_lengths = batch

        # Forward pass
        preds = self(padded_seqs, static_features, seq_lengths)

        # Вычисляем loss
        if hasattr(self.criterion, "forward") and "horizon_hours" in self.criterion.forward.__code__.co_varnames:
            # Используем loss функцию, которая поддерживает горизонты прогноза
            loss = self.criterion(preds, targets, horizon_hours=horizon_hours, sample_weight=weights)
        elif self._criterion_handles_weights:
            loss = self.criterion(preds, targets, sample_weight=weights)
        else:
            loss_tensor = self.criterion(preds, targets)
            if loss_tensor.dim() > 1:
                loss_tensor = loss_tensor.mean(dim=1)
            loss = (loss_tensor * weights).mean()

        # Log the learnable alpha parameter
        if hasattr(self.criterion, "learnable_norm") and hasattr(self.criterion.learnable_norm, "alpha"):
            alpha_value = self.criterion.learnable_norm.alpha
            if isinstance(alpha_value, (int, float)):
                self.log(f"{stage}_horizon_alpha", alpha_value, prog_bar=False, on_epoch=True)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss  # type: ignore[no-any-return]

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def on_train_epoch_end(self) -> None:
        """Вычисляет дополнительные метрики в конце эпохи тренировки."""
        # Вычисляем метрику для 48-часового прогноза на тренировочных данных
        self._compute_train_48h_metric()

    def _compute_train_48h_metric(self) -> None:
        """Вычисляет метрику попадания в 300 км сектор для 48-часового прогноза на тренировочных данных."""
        try:
            # Получаем тренировочные данные из trainer
            if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
                train_loader = self.trainer.datamodule.train_dataloader()

                # Собираем все предсказания и цели
                all_preds: list[torch.Tensor] = []
                all_targets: list[torch.Tensor] = []
                all_horizons: list[float] = []
                all_sequences: list[torch.Tensor] = []

                # Сохраняем текущий режим модели
                was_training = self.training

                self.eval()
                with torch.no_grad():
                    for batch in train_loader:
                        padded_seqs, static_features, targets, weights, horizon_hours, seq_lengths = batch

                        preds = self(
                            padded_seqs.to(self.device), static_features.to(self.device), seq_lengths.to(self.device)
                        )

                        batch_size = preds.shape[0]
                        for i in range(batch_size):
                            all_preds.append(preds[i].cpu())
                            all_targets.append(targets[i])

                            # Извлекаем горизонт из отдельного тензора
                            horizon = horizon_hours[i].cpu().item()
                            all_horizons.append(horizon)

                            # Сохраняем последовательности для извлечения текущих координат
                            all_sequences.append(padded_seqs[i])

                # Объединяем все образцы
                all_preds_tensor = torch.stack(all_preds, dim=0)
                all_targets_tensor = torch.stack(all_targets, dim=0)
                all_horizons_tensor = torch.tensor(all_horizons, dtype=torch.float32)

                # Фильтруем только 48-часовые прогнозы
                mask_48h = all_horizons_tensor == 48.0

                if mask_48h.sum() > 0:
                    preds_48h = all_preds_tensor[mask_48h]
                    targets_48h = all_targets_tensor[mask_48h]
                    sequences_48h = [all_sequences[i] for i in range(len(all_sequences)) if mask_48h[i].item()]

                    # Вычисляем точные ошибки в километрах используя haversine distance
                    errors_km = self._compute_haversine_errors_list(sequences_48h, targets_48h, preds_48h)

                    # Вычисляем процент попадания в 300 км сектор
                    p300_48h = (errors_km < 300.0).float().mean() * 100.0

                    # Логируем метрику с отображением в прогресс-баре
                    self.log("train_p300_48h", p300_48h, prog_bar=True, on_epoch=True)
                else:
                    # Если нет 48-часовых прогнозов, логируем 0
                    self.log("train_p300_48h", 0.0, prog_bar=True, on_epoch=True)

                # Восстанавливаем исходный режим модели
                if was_training:
                    self.train()

        except Exception as e:
            # Если что-то пошло не так, логируем ошибку но не прерываем обучение
            print(f"Warning: Could not compute train 48h metric: {e}")
            # Логируем 0 в случае ошибки
            self.log("train_p300_48h", 0.0, prog_bar=True, on_epoch=True)
            # Восстанавливаем режим обучения в случае ошибки
            if was_training:
                self.train()

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        """Вычисляет дополнительные метрики в конце эпохи валидации."""
        # Вычисляем метрику для 48-часового прогноза
        self._compute_48h_metric()

    def _compute_48h_metric(self) -> None:
        """Вычисляет метрику попадания в 300 км сектор для 48-часового прогноза."""
        try:
            # Получаем валидационные данные из trainer
            if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
                val_loader = self.trainer.datamodule.val_dataloader()

                # Собираем все предсказания и цели
                all_preds: list[torch.Tensor] = []
                all_targets: list[torch.Tensor] = []
                all_horizons: list[float] = []
                all_sequences: list[torch.Tensor] = []

                # Сохраняем текущий режим модели
                was_training = self.training

                self.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        padded_seqs, static_features, targets, weights, horizon_hours, seq_lengths = batch

                        # Получаем предсказания
                        preds = self(
                            padded_seqs.to(self.device), static_features.to(self.device), seq_lengths.to(self.device)
                        )

                        # Обрабатываем каждый образец отдельно, чтобы избежать проблем с размерами
                        batch_size = preds.shape[0]
                        for i in range(batch_size):
                            all_preds.append(preds[i].cpu())
                            all_targets.append(targets[i])

                            # Извлекаем горизонт из отдельного тензора
                            horizon = horizon_hours[i].cpu().item()
                            all_horizons.append(horizon)

                            # Сохраняем последовательности для извлечения текущих координат
                            all_sequences.append(padded_seqs[i])

                # Объединяем все образцы
                all_preds_tensor = torch.stack(all_preds, dim=0)
                all_targets_tensor = torch.stack(all_targets, dim=0)
                all_horizons_tensor = torch.tensor(all_horizons, dtype=torch.float32)

                mask_48h = all_horizons_tensor == 48.0

                if torch.sum(mask_48h) > 0:
                    preds_48h = all_preds_tensor[mask_48h]
                    targets_48h = all_targets_tensor[mask_48h]
                    sequences_48h = [all_sequences[i] for i in range(len(all_sequences)) if mask_48h[i].item()]

                    # Вычисляем точные ошибки в километрах используя haversine distance
                    errors_km = self._compute_haversine_errors_list(sequences_48h, targets_48h, preds_48h)

                    # Вычисляем процент попадания в 300 км сектор
                    p300_48h = (errors_km < 300.0).float().mean() * 100.0

                    # Логируем метрику с отображением в прогресс-баре
                    self.log("val_p300_48h", p300_48h, prog_bar=True, on_epoch=True)
                else:
                    # Если нет 48-часовых прогнозов, логируем 0
                    self.log("val_p300_48h", 0.0, prog_bar=True, on_epoch=True)

                # Восстанавливаем исходный режим модели
                if was_training:
                    self.train()

        except Exception as e:
            # Если что-то пошло не так, логируем ошибку но не прерываем обучение
            print(f"Warning: Could not compute 48h metric: {e}")
            # Логируем 0 в случае ошибки
            self.log("val_p300_48h", 0.0, prog_bar=True, on_epoch=True)
            # Восстанавливаем режим обучения в случае ошибки
            if was_training:
                self.train()

    def _compute_haversine_errors(
        self, sequences: torch.Tensor, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.Tensor:
        """Вычисляет ошибки в километрах используя haversine distance."""
        coord_processor = CoordinateProcessor()
        errors_km = []

        for i in range(len(sequences)):
            # Извлекаем текущие координаты из последовательности
            curr_lat, curr_lon = self._extract_current_coords(sequences[i])

            # Вычисляем предсказанные координаты
            pred_lat = curr_lat + predictions[i, 0].item()
            pred_lon = curr_lon + predictions[i, 1].item()

            # Истинные координаты
            true_lat = curr_lat + targets[i, 0].item()
            true_lon = curr_lon + targets[i, 1].item()

            # Вычисляем haversine distance
            error_km = coord_processor.haversine_distance(true_lat, true_lon, pred_lat, pred_lon)
            errors_km.append(error_km)

        return torch.tensor(errors_km, dtype=torch.float32)

    def _compute_haversine_errors_list(
        self, sequences: list[torch.Tensor], targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.Tensor:
        """Вычисляет ошибки в километрах используя haversine distance для списка последовательностей."""
        coord_processor = CoordinateProcessor()
        errors_km = []

        for i in range(len(sequences)):
            # Извлекаем текущие координаты из последовательности
            curr_lat, curr_lon = self._extract_current_coords(sequences[i])

            # Вычисляем предсказанные координаты
            pred_lat = curr_lat + predictions[i, 0].item()
            pred_lon = curr_lon + predictions[i, 1].item()

            # Истинные координаты
            true_lat = curr_lat + targets[i, 0].item()
            true_lon = curr_lon + targets[i, 1].item()

            # Вычисляем haversine distance
            error_km = coord_processor.haversine_distance(true_lat, true_lon, pred_lat, pred_lon)
            errors_km.append(error_km)

        return torch.tensor(errors_km, dtype=torch.float32)

    def _extract_current_coords(self, seq: torch.Tensor) -> tuple[float, float]:
        """Извлекает текущие координаты из последней непустой строки последовательности."""
        # Находим последнюю непустую строку
        non_zero_mask = (seq != 0).any(dim=1)
        if non_zero_mask.any():
            last_idx = int(torch.where(non_zero_mask)[0][-1])
            return float(seq[last_idx, 0]), float(seq[last_idx, 1])
        return 0.0, 0.0

    def configure_optimizers(self) -> dict[str, Any]:  # type: ignore[override]
        optimiser = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=3)
        return {
            "optimizer": optimiser,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss_epoch",
                "interval": "epoch",
                "frequency": 1,
            },
        }
