"""
This module provides the necessary classes for text classification models.

Classes:
- TextClassifier: A class representing a text classifier model.
- DatasetConfig: A class representing the configuration for the dataset used in training.
- TrainingConfig: A class representing the configuration for the training process.
- CrossValidateConfig: A class representing the configuration for cross-validation.

Usage:
Import the desired class from this module to use it in your code.
"""

from .text_classifier import (
    CrossValidateConfig,
    DatasetConfig,
    TextClassifier,
    TrainingConfig,
)

__all__ = [
    "TextClassifier",
    "DatasetConfig",
    "TrainingConfig",
    "CrossValidateConfig",
]
