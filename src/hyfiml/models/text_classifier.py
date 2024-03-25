"""
Module: text_classifier.py

This module provides a TextClassifier class for performing text classification tasks using transformer models from the
Hugging Face library. The TextClassifier class allows loading datasets, preprocessing data, training models, making
predictions, and identifying potential label errors.

Classes:
    - TrainingConfig: A Pydantic BaseModel class for configuring training arguments.
    - DatasetConfig: A Pydantic BaseModel class for configuring dataset arguments.
    - CrossValidateConfig: A Pydantic BaseModel class for configuring cross-validation arguments.
    - TextClassifier: The main class for text classification tasks.

The TextClassifier class includes the following methods:
    - load_dataset: Loads a dataset using the configuration specified in dataset_config.
    - preprocess_dataset: Preprocesses the dataset by tokenizing the text and converting labels.
    - split_dataset: Splits the dataset into train, test, and optionally dev sets based on the configuration specified in dataset_config.
    - compute_metrics: Computes evaluation metrics during training.
    - train: Trains the model on the provided dataset using the specified training configuration.
    - predict: Makes predictions on a new dataset using the trained model.
    - save_model: Saves the trained model to a specified directory.
    - load_model: Loads a trained model from a specified directory.
    - plot_confusion_matrix: Plots the confusion matrix for a given dataset.
    - cross_validate_and_predict: Performs cross-validation and prediction using the trained model.
    - find_potential_label_errors: Finds potential label errors using cleanlab's find_label_issues function.

The TextClassifier class takes the following arguments during initialization:
    - model_name: The name of the transformer model to use.
    - num_labels: The number of labels in the classification task.
    - dataset_config: An instance of the DatasetConfig class specifying the dataset configuration.
    - training_config: An instance of the TrainingConfig class specifying the training configuration.
    - cross_validate_config: An instance of the CrossValidateConfig class specifying the cross-validation configuration.

Example usage:
    # Create dataset, training, and cross-validation configurations
    dataset_config = DatasetConfig(
        dataset_name="imdb",
        text_column_name="text",
        label_column_name="label",
        num_labels=2,
    )

    training_config = TrainingConfig(
        output_dir="output",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
    )

    cross_validate_config = CrossValidateConfig(
        n_splits=5,
        validation_size=0.1,
        random_state=42,
        shuffle=True,
    )

    # Create a TextClassifier instance
    classifier = TextClassifier(
        model_name="bert-base-uncased",
        dataset_config=dataset_config,
        training_config=training_config,
        cross_validate_config=cross_validate_config,
    )

    # Load the dataset
    dataset = classifier.load_dataset()

    # Train the model
    classifier.train(dataset)

    # Make predictions on a new dataset
    new_dataset = load_dataset("imdb", split="test")
    predictions = classifier.predict(new_dataset)

    # Perform cross-validation and find potential label errors
    predictions = classifier.cross_validate_and_predict(dataset)
    label_issues_info = classifier.find_potential_label_errors(predictions, dataset["label"])
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import evaluate
import matplotlib.pyplot as plt
import numpy as np
from cleanlab.filter import find_label_issues
from datasets import Dataset, DatasetDict, load_dataset
from hyfi.composer import BaseModel, field_validator
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import KFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)


class TrainingConfig(BaseModel):
    """
    Configuration for training arguments.

    Attributes:
        output_dir (str): The output directory where the model predictions and checkpoints will be written.
        num_train_epochs (int): The number of training epochs to perform.
        per_device_train_batch_size (int): The batch size per GPU/CPU for training. Default is 8.
        per_device_eval_batch_size (int): The batch size per GPU/CPU for evaluation. Default is 8.
        warmup_steps (int): The number of steps for the warmup phase during training. Default is 500.
        weight_decay (float): The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in AdamW optimizer. Default is 0.01.
        learning_rate (float): The learning rate to use during training. Default is 2e-5.
        logging_dir (str): The directory to save the logs. Default is "logs".
        logging_steps (int): The logging steps. Default is 10.
        evaluation_strategy (str): The evaluation strategy to adopt during training. Default is "epoch".
        save_strategy (str): The checkpoint save strategy to adopt during training. Default is "epoch".
        load_best_model_at_end (bool): Whether to load the best model found during training at the end of training. Default is True.
        metric_for_best_model (str): The metric to use to compare two different models. Default is "accuracy".
    """

    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    warmup_steps: int = 500
    weight_decay: float = 0.01
    learning_rate: float = 2e-5
    logging_dir: str = "logs"
    logging_steps: int = 10
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_accuracy"


class DatasetConfig(BaseModel):
    """
    Configuration for dataset arguments.

    Attributes:
        dataset_name (str): The name of the dataset to load from the Hugging Face datasets library.
        dataset_config_name (Optional[str]): The configuration name of the dataset. Default is None.
        data_dir (Optional[str]): The directory containing the dataset files. Default is None.
        data_files (Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]): The dataset files to load. Default is None.
        train_split_name (str): The name of the train split. Default is "train".
        test_split_name (str): The name of the test split. Default is "test".
        text_column_name (str): The name of the column containing the text data. Default is "text".
        label_column_name (str): The name of the column containing the label data. Default is "label".
        load_data_split (str): The split of the dataset to load. Default is "train".
        test_size (float): The proportion of the dataset to include in the test split. Default is 0.2.
        stratify_on (Optional[str]): The column to use for stratified splitting. Default is None.
        random_state (Optional[int]): The random state for reproducibility. Default is None.
        max_length (int): The maximum length of the text sequences to be processed. Default is 256.
        num_labels (Optional[int]): The number of labels in the classification task. Default is None.
        id2label (Optional[Dict[int, str]]): A dictionary mapping label indices to label names. Default is None.
    """

    dataset_name: str
    dataset_config_name: Optional[str] = None
    data_dir: Optional[str] = None
    data_files: Optional[
        Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]
    ] = None
    train_split_name: str = "train"
    test_split_name: str = "test"
    text_column_name: str = "text"
    label_column_name: str = "label"
    load_data_split: str = "train"
    test_size: float = 0.2
    stratify_on: Optional[str] = None
    random_state: Optional[int] = None
    max_length: int = 256
    num_labels: Optional[int] = None
    id2label: Optional[Dict[int, str]] = None


class CrossValidateConfig(BaseModel):
    """
    Configuration for cross-validation arguments.

    Attributes:
        n_splits (int): The number of splits for cross-validation. Default is 5.
        validation_size (float): The proportion of the training set to include in the validation split. Default is 0.1.
        random_state (int): The random state for reproducibility. Default is 42.
        shuffle (bool): Whether to shuffle the data before splitting. Default is True.
    """

    n_splits: int = 5
    validation_size: float = 0.1
    random_state: int = 42
    shuffle: bool = True


class TextClassifier(BaseModel):
    """
    Text classifier based on transformer models from Hugging Face.

    Attributes:
        model_name (str): The name of the transformer model to use.
        dataset_config (DatasetConfig): The configuration for the dataset.
        training_config (TrainingConfig): The configuration for training.
        cross_validate_config (CrossValidateConfig): The configuration for cross-validation.
        tokenizer (AutoTokenizer): The tokenizer for the transformer model. Automatically initialized.
        model (AutoModelForSequenceClassification): The transformer model for sequence classification. Automatically initialized.
    """

    model_name: str
    dataset_config: DatasetConfig
    training_config: TrainingConfig
    cross_validate_config: CrossValidateConfig

    __tokenizer__: Optional[AutoTokenizer] = None
    __model__: Optional[AutoModelForSequenceClassification] = None
    __label2id__: Optional[Dict[str, int]] = None

    @property
    def tokenizer(self):
        if self.__tokenizer__ is None:
            self.__tokenizer__ = AutoTokenizer.from_pretrained(self.model_name)
        return self.__tokenizer__

    @property
    def label2id(self):
        # convert id2label to label2id
        if self.__label2id__ is None and self.dataset_config.id2label is not None:
            self.__label2id__ = {v: k for k, v in self.dataset_config.id2label.items()}
        return self.__label2id__

    @property
    def num_labels(self):
        if self.dataset_config.id2label is not None:
            return len(self.dataset_config.id2label)
        if self.dataset_config.num_labels is None:
            raise ValueError(
                "num_labels must be provided if id2label is not specified."
            )
        return self.dataset_config.num_labels

    @property
    def model(self):
        if self.__model__ is None:
            self.__model__ = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
            )
        return self.__model__

    def load_dataset(self) -> Dataset:
        """
        Load the dataset using the configuration specified in dataset_config.

        Returns:
            Dataset: The loaded dataset.
        """
        return load_dataset(
            self.dataset_config.dataset_name,
            name=self.dataset_config.dataset_config_name,
            data_dir=self.dataset_config.data_dir,
            data_files=self.dataset_config.data_files,
            split=self.dataset_config.load_data_split,
        )

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """
        Preprocess the dataset by tokenizing the text and converting labels.

        Args:
            dataset (Dataset): The dataset to preprocess.

        Returns:
            Dataset: The preprocessed dataset.
        """

        def tokenize(examples):
            return self.tokenizer(
                examples[self.dataset_config.text_column_name],
                padding="max_length",
                truncation=True,
                max_length=512,
            )

        def convert_labels(examples):
            if self.label2id is not None:
                labels = [self.label2id[label] for label in examples["original_labels"]]
            else:
                labels = examples["original_labels"]
            return {"labels": labels}

        # Tokenize text
        dataset = dataset.map(tokenize, batched=True)

        # Convert labels
        dataset = dataset.rename_column(
            self.dataset_config.label_column_name, "original_labels"
        )
        dataset = dataset.map(convert_labels, batched=True)

        # Remove unnecessary columns
        dataset = dataset.remove_columns(
            [
                self.dataset_config.text_column_name,
                "original_labels",
            ]
        )

        # dataset.set_format("torch")
        return dataset

    def split_dataset(self, dataset: Dataset) -> DatasetDict:
        """
        Split the dataset into train, test, and optionally dev sets based on the configuration specified in dataset_config.

        Args:
            dataset (Dataset): The dataset to split.

        Returns:
            DatasetDict: A dictionary containing the train, test, and optionally dev datasets.
        """
        dataset_dict = dataset.train_test_split(
            test_size=self.dataset_config.test_size,
            stratify_by_column=self.dataset_config.stratify_on,
            seed=self.dataset_config.random_state,
        )
        dataset_dict = DatasetDict(
            {
                self.dataset_config.train_split_name: dataset_dict["train"],
                self.dataset_config.test_split_name: dataset_dict["test"],
            }
        )
        return dataset_dict

    def compute_metrics(self, pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute the evaluation metrics.

        Args:
            pred (EvalPrediction): The predictions and labels.

        Returns:
            float: The accuracy of the predictions.
        """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        accuracy = evaluate.load("accuracy")
        return accuracy.compute(predictions=preds, references=labels)

    def train(
        self,
        dataset: Dataset,
        training_config: Optional[TrainingConfig] = None,
    ) -> None:
        """
        Train the model on the provided dataset.

        Args:
            dataset (Dataset): The dataset to use for training.
            training_config (Optional[TrainingConfig]): The training configuration to use. If not provided, the default training_config will be used.
        """
        dataset = self.preprocess_dataset(dataset)
        dataset_dict = self.split_dataset(dataset)
        train_dataset = dataset_dict[self.dataset_config.train_split_name]
        test_dataset = dataset_dict[self.dataset_config.test_split_name]

        if training_config is None:
            training_config = self.training_config

        training_args_dict = training_config.model_dump()
        training_args_dict["output_dir"] = training_config.output_dir
        training_args = TrainingArguments(**training_args_dict)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

    def predict(self, dataset: Dataset) -> List[int]:
        """
        Make predictions on the provided dataset.

        Args:
            dataset (Dataset): The dataset to make predictions on.

        Returns:
            List[int]: The predicted labels.
        """
        dataset = self.preprocess_dataset(dataset)
        trainer = Trainer(model=self.model)
        predictions = trainer.predict(dataset)
        preds = predictions.predictions.argmax(-1)
        return [self.dataset_config.id2label[label] for label in preds.tolist()]

    def save_model(self, output_dir: str) -> None:
        """
        Save the trained model.

        Args:
            output_dir (str): The directory to save the model.
        """
        self.model.save_pretrained(output_dir)

    def load_model(self, model_dir: str) -> None:
        """
        Load a trained model.

        Args:
            model_dir (str): The directory containing the trained model.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    def plot_confusion_matrix(self, dataset: Dataset, labels: List[str]) -> None:
        """
        Plot the confusion matrix for the provided dataset.

        Args:
            dataset (Dataset): The dataset to evaluate.
            labels (List[str]): The list of labels.
        """
        dataset = self.preprocess_dataset(dataset)
        trainer = Trainer(model=self.model)
        predictions = trainer.predict(dataset)
        preds = predictions.predictions.argmax(-1)
        true_labels = dataset["labels"]

        cm = confusion_matrix(true_labels, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues")
        plt.show()

    def cross_validate_and_predict(
        self,
        dataset: Dataset,
        training_config: Optional[TrainingConfig] = None,
        cross_validate_config: Optional[CrossValidateConfig] = None,
    ) -> List[List[int]]:
        """
        Perform cross-validation and prediction using the trained model.

        Args:
            dataset (Dataset): The dataset to use for cross-validation and prediction.
            training_config (Optional[TrainingConfig]): The training configuration to use. If not provided, the default training_config will be used.
            cross_validate_config (Optional[CrossValidateConfig]): The cross-validation configuration to use. If not provided, the default cross_validate_config will be used.

        Returns:
            List[List[int]]: A list of predicted labels for each fold.
        """
        dataset = self.preprocess_dataset(dataset)

        if cross_validate_config is None:
            cross_validate_config = self.cross_validate_config

        kf = KFold(
            n_splits=cross_validate_config.n_splits,
            shuffle=cross_validate_config.shuffle,
            random_state=cross_validate_config.random_state,
        )

        predictions_list = []
        for train_index, test_index in kf.split(dataset):
            train_dataset = dataset.select(train_index)
            test_dataset = dataset.select(test_index)

            if cross_validate_config.validation_size > 0:
                train_dataset, validation_dataset = train_dataset.train_test_split(
                    test_size=cross_validate_config.validation_size,
                    shuffle=cross_validate_config.shuffle,
                    seed=cross_validate_config.random_state,
                ).values()
            else:
                validation_dataset = None

            if training_config is None:
                training_config = self.training_config

            training_args_dict = training_config.dict()
            training_args_dict["output_dir"] = f"output_{len(predictions_list)}"
            training_args = TrainingArguments(**training_args_dict)

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                compute_metrics=self.compute_metrics,
            )

            trainer.train()
            predictions = trainer.predict(test_dataset)
            preds = predictions.predictions.argmax(-1)
            predictions_list.append(preds.tolist())

        return predictions_list

    def find_potential_label_errors(
        self, predictions_list: List[List[int]], labels: List[int]
    ) -> Dict[str, Any]:
        """
        Find potential label errors using cleanlab's find_label_issues function.

        Args:
            predictions_list (List[List[int]]): A list of predicted labels for each fold.
            labels (List[int]): The true labels of the samples.

        Returns:
            Dict[str, Any]: A dictionary containing information about the identified label issues.
        """
        num_folds = len(predictions_list)
        num_samples = len(labels)

        predicted_probs = np.zeros((num_samples, self.num_labels))

        for fold_predictions in predictions_list:
            for idx, label in enumerate(fold_predictions):
                predicted_probs[idx][label] += 1 / num_folds

        if self.label2id is not None:
            labels = [self.label2id[label] for label in labels]

        return find_label_issues(
            labels,
            predicted_probs,
            return_indices_ranked_by="self_confidence",
        )
