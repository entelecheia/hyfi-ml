from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score

# Load dataset
dataset = load_dataset("imdb")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("entelecheia/ekonbert-base")
model = AutoModelForSequenceClassification.from_pretrained(
    "entelecheia/ekonbert-base", num_labels=2
)


# Tokenize data
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=512)


tokenized_datasets = dataset.map(tokenize, batched=True, remove_columns=["text"])

# Add labels to tokenized datasets
tokenized_datasets = tokenized_datasets.map(
    lambda example: {"labels": example["label"]}, remove_columns=["label"]
)


# Define compute_metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}


# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()
