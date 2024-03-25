from hyfiml import HyFI
from hyfiml.models import (
    CrossValidateConfig,
    DatasetConfig,
    TextClassifier,
    TrainingConfig,
)

h = HyFI.initialize(
    project_name="hyfi-ml",
    project_root="$HOME/workspace/projects/hyfi-ml",
    logging_level="INFO",
    verbose=True,
)

print("project_dir:", h.project.root_dir)
print("project_workspace_dir:", h.project.workspace_dir)

data_dir = h.project.workspace_dir / "data"

data_file = (
    "/home/yjlee/workspace/projects/hyfi-ml/workspace/data/esg_polarity.parquet",
)
data = HyFI.load_dataframes(data_file)
# Find the rows with missing values in the 'labels' column
print(data[data.labels.isnull()])
print(data.labels.unique())
print(data.head())

dataset_config = DatasetConfig(
    dataset_name="parquet",
    data_files=data_file,
    train_split_name="train",
    test_split_name="test",
    text_column_name="text",
    label_column_name="labels",
    load_data_split="train",
    test_size=0.1,
    dev_size=None,
    stratify_on=None,
    random_state=None,
    max_length=64,
    num_labels=3,
    id2label={0: "Negative", 1: "Neutral", 2: "Positive"},
)

training_config = TrainingConfig(
    output_dir="output",
    num_train_epochs=3,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_steps=50,
    weight_decay=0.01,
    learning_rate=2e-5,
    logging_dir="logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

cross_validate_config = CrossValidateConfig(
    n_splits=5,
    validation_size=0.1,
    random_state=42,
    shuffle=True,
)

classifier = TextClassifier(
    model_name="entelecheia/ekonbert-base",
    dataset_config=dataset_config,
    training_config=training_config,
    cross_validate_config=cross_validate_config,
)

print(classifier.label2id, classifier.dataset_config.id2label)
dataset = classifier.load_dataset()
classifier.train(dataset)
