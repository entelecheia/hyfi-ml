"""
Module: text_classifier.py

This module provides a TextClassifier class for performing text classification tasks using transformer models from the Hugging Face library.
The TextClassifier class allows loading datasets, preprocessing data, training models, making predictions, and identifying potential label errors.

Classes:
    - TrainingConfig: A Pydantic BaseModel class for configuring training arguments.
    - TextClassifier: The main class for text classification tasks.

The TextClassifier class includes the following methods:
    - load_dataset: Loads a dataset from the Hugging Face datasets library.
    - preprocess_dataset: Preprocesses the dataset by tokenizing the text and converting labels.
    - split_dataset: Splits the dataset into train and test sets.
    - compute_metrics: Computes evaluation metrics during training.
    - train: Trains the model on the provided dataset using the specified training arguments.
    - predict: Makes predictions on a new dataset using the trained model.
    - save_model: Saves the trained model to a specified directory.
    - load_model: Loads a trained model from a specified directory.
    - plot_confusion_matrix: Plots the confusion matrix for a given dataset.
    - cross_validate_and_predict: Performs cross-validation and prediction using the trained model.
    - find_potential_label_errors: Finds potential label errors using cleanlab's find_label_issues function.

The module also includes the following dependencies:
    - typing: Provides type hinting support.
    - datasets: Hugging Face datasets library for loading and manipulating datasets.
    - pydantic: Library for data validation and settings management using Python type annotations.
    - transformers: Hugging Face transformers library for natural language processing tasks.
    - numpy: Library for numerical operations.
    - evaluate: Hugging Face evaluate library for computing evaluation metrics.
    - sklearn.metrics: Scikit-learn library for computing confusion matrix and other metrics.
    - matplotlib: Library for data visualization.
    - sklearn.model_selection: Scikit-learn library for model selection and cross-validation.
    - cleanlab.filter: Cleanlab library for identifying potential label errors.

Example usage:
    # Create a TextClassifier object
    classifier = TextClassifier(
        model_name="bert-base-uncased",
        num_labels=2,
        dataset_name="imdb",
    )

    # Load the dataset
    dataset = classifier.load_dataset()

    # Configure training arguments
    training_args = TrainingConfig(output_dir="output", num_train_epochs=3)

    # Perform cross-validation and prediction
    predictions = classifier.cross_validate_and_predict(
        dataset,
        training_args=training_args,
        n_splits=5,
        validation_size=0.1,
        random_state=42,
        shuffle=True,
    )

    # Find potential label errors
    true_labels = dataset["label"]
    label_issues_info = classifier.find_potential_label_errors(predictions, true_labels)

    # Print the indices of samples with potential label errors
    print("Indices of samples with potential label errors:")
    print(label_issues_info["indices"])
"""

from typing import Dict, List, Optional

import evaluate
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict, load_dataset
from pydantic import BaseModel, Field
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
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
    logging_dir: str = "logs"
    logging_steps: int = 10
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"


class TextClassifier(BaseModel):
    """
    Text classifier based on transformer models from Hugging Face.

    Attributes:
        model_name (str): The name of the transformer model to use.
        num_labels (int): The number of labels for classification.
        dataset_name (str): The name of the dataset to use.
        dataset_config_name (str, optional): The configuration name of the dataset. Default is None.
        dataset_split (str): The split of the dataset to use. Default is "train".
        train_split_name (str): The name of the train split. Default is "train".
        test_split_name (str): The name of the test split. Default is "test".
        text_column_name (str): The name of the column containing the text data. Default is "text".
        label_column_name (str): The name of the column containing the label data. Default is "label".
        tokenizer (AutoTokenizer): The tokenizer for the transformer model. Automatically initialized.
        model (AutoModelForSequenceClassification): The transformer model for sequence classification. Automatically initialized.
    """

    model_name: str
    num_labels: int
    dataset_name: str
    dataset_config_name: Optional[str] = None
    dataset_split: str = "train"
    train_split_name: str = "train"
    test_split_name: str = "test"
    text_column_name: str = "text"
    label_column_name: str = "label"
    tokenizer: AutoTokenizer = Field(init=False)
    model: AutoModelForSequenceClassification = Field(init=False)

    def __init__(self, **data):
        super().__init__(**data)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )

    def load_dataset(self) -> Dataset:
        """
        Load the dataset.

        Returns:
            Dataset: The loaded dataset.
        """
        return load_dataset(
            self.dataset_name,
            self.dataset_config_name,
            split=self.dataset_split,
        )

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """
        Preprocess the dataset by tokenizing the text and converting the labels.

        Args:
            dataset (Dataset): The dataset to preprocess.

        Returns:
            Dataset: The preprocessed dataset.
        """

        def tokenize(examples):
            return self.tokenizer(examples[self.text_column_name], truncation=True)

        dataset = dataset.map(tokenize, batched=True)
        dataset = dataset.map(
            lambda examples: {"labels": examples[self.label_column_name]}
        )
        dataset = dataset.remove_columns(
            [self.text_column_name, self.label_column_name]
        )
        dataset.set_format("torch")
        return dataset

    def split_dataset(
        self,
        dataset: Dataset,
        test_size: float = 0.2,
        dev_size: Optional[float] = None,
        stratify_on: Optional[str] = None,
        random_state: Optional[int] = None,
    ) -> DatasetDict:
        """
        Split the dataset into train, test, and optionally dev sets.

        Args:
            dataset (Dataset): The dataset to split.
            test_size (float): The proportion of the dataset to include in the test split. Default is 0.2.
            dev_size (float, optional): The proportion of the dataset to include in the dev split. Default is None.
            stratify_on (str, optional): The column to use for stratified splitting. Default is None.
            random_state (int, optional): The random state for reproducibility. Default is None.

        Returns:
            DatasetDict: A dictionary containing the train, test, and optionally dev datasets.
        """
        if dev_size is None:
            # Split the dataset into train and test sets
            dataset_dict = dataset.train_test_split(
                test_size=test_size,
                stratify_by_column=stratify_on,
                seed=random_state,
            )
            dataset_dict = DatasetDict(
                {
                    self.train_split_name: dataset_dict["train"],
                    self.test_split_name: dataset_dict["test"],
                }
            )
        else:
            # Split the dataset into train, test, and dev sets
            train_test_dict = dataset.train_test_split(
                test_size=test_size + dev_size,
                stratify_by_column=stratify_on,
                seed=random_state,
            )
            test_dev_dict = train_test_dict["test"].train_test_split(
                test_size=dev_size / (test_size + dev_size),
                stratify_by_column=stratify_on,
                seed=random_state,
            )
            dataset_dict = DatasetDict(
                {
                    self.train_split_name: train_test_dict["train"],
                    self.test_split_name: test_dev_dict["train"],
                    "dev": test_dev_dict["test"],
                }
            )

        return dataset_dict

    def compute_metrics(self, pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute the evaluation metrics.

        Args:
            pred (EvalPrediction): The predictions and labels.

        Returns:
            Dict[str, float]: A dictionary containing the computed metrics.
        """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        accuracy = evaluate.load("accuracy")
        return {"accuracy": accuracy.compute(predictions=preds, references=labels)}

    def train(self, dataset: Dataset, training_args: TrainingConfig) -> None:
        """
        Train the model on the given dataset.

        Args:
            dataset (Dataset): The dataset to use for training.
            training_args (TrainingConfig): The training configuration.
        """
        dataset = self.preprocess_dataset(dataset)
        dataset_dict = self.split_dataset(dataset)
        train_dataset = dataset_dict[self.train_split_name]
        test_dataset = dataset_dict[self.test_split_name]

        training_args_dict = training_args.dict()
        training_args_dict["output_dir"] = training_args.output_dir
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
        Make predictions on the given dataset.

        Args:
            dataset (Dataset): The dataset to make predictions on.

        Returns:
            List[int]: The predicted labels.
        """
        dataset = self.preprocess_dataset(dataset)
        trainer = Trainer(model=self.model)
        predictions = trainer.predict(dataset)
        preds = predictions.predictions.argmax(-1)
        return preds.tolist()

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
        Plot the confusion matrix for the given dataset.

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
