"""
Модуль для обработки данных о циклонах.

Упрощенная архитектура, использующая возможности FeatureRegistry
для централизованного управления фичами и их вычислениями.
"""

from .base_processor import BaseDataProcessor
from .data_processor import DataProcessor
from .dataset_models import ProcessedDataset, SequenceConfig
from .inference_processor import InferenceDataProcessor

__all__ = [
    "BaseDataProcessor",
    "DataProcessor",
    "InferenceDataProcessor",
    "ProcessedDataset",
    "SequenceConfig",
]
