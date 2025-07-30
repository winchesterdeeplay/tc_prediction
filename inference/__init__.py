"""
Модуль для инференса моделей предсказания траекторий циклонов.

Этот модуль предоставляет оптимизированные пайплайны для инференса
с использованием ONNX Runtime.
"""

from .onnx_pipeline import ONNXInferencePipeline, ONNXInferencePipelineFactory

__all__ = ["ONNXInferencePipeline", "ONNXInferencePipelineFactory"]
