import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
import os

# Select device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
df = pd.read_csv("cleaned_dataset.csv")
label_mapping = {"real": 0, "fake": 1}
df["label"] = df["label"].map(label_mapping)

dataset = Dataset.from_pandas(df)
dataset_split = dataset.train_test_split(test_size=0.2, seed=42)
train_data = dataset_split["train"]
test_data = dataset_split["test"]

# Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_batch(batch):
    return tokenizer(
        batch["statement"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_data = train_data.map(tokenize_batch, batched=True)
test_data = test_data.map(tokenize_batch, batched=True)

train_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

def compute_accuracy(p):
    labels = p.label_ids
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": (preds == labels).mean()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer,
    compute_metrics=compute_accuracy,
)

# Resume checkpoint if available
last_ckpt = None
if os.path.isdir("./results"):
    ckpt_dirs = [os.path.join("./results", d) for d in os.listdir("./results") if d.startswith("checkpoint")]
    if ckpt_dirs:
        last_ckpt = sorted(ckpt_dirs)[-1]
        print(f"Resuming training from checkpoint: {last_ckpt}")

trainer.train(resume_from_checkpoint=last_ckpt)
eval_metrics = trainer.evaluate()
print("Evaluation metrics:", eval_metrics)

model.save_pretrained("./models/distilbert_model")
tokenizer.save_pretrained("./models/distilbert_model")
print("Model and tokenizer saved to ./models/distilbert_model")
