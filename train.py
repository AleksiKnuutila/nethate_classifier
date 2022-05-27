import pandas as pd
import numpy as np
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset, load_metric


def tokenize_function(string):
    return tokenizer(string["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def predict(examples):
    return {"predictions": finetuned_finbert(examples["text"], truncation=True)}


# Only tweet IDs are published, so this file is not part of repository.
# The annotations can in part be recreated by "hydrating" tweets.
df = pd.read_csv("annotations_with_text.csv")

df["text"] = df["text"].str.encode("utf-8", errors="ignore").astype(str)
df["labels"] = df["labels"].astype(int)
train_ds = Dataset.from_pandas(df[["text", "labels"]])

train_ds = train_ds.train_test_split(test_size=0.2)
tokenizer = AutoTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-uncased-v1")
train_dataset = train_ds["train"].map(tokenize_function, batched=True).shuffle(seed=1)
eval_dataset = train_ds["test"].map(tokenize_function, batched=True).shuffle(seed=1)

finnish_classifier = pipeline(
    model="TurkuNLP/bert-base-finnish-uncased-v1",
    task="text-classification",
    return_all_scores=True,
)

model = AutoModelForSequenceClassification.from_pretrained(
    "TurkuNLP/bert-base-finnish-uncased-v1"
)

training_args = TrainingArguments(
    "nethate", evaluation_strategy="epoch", logging_steps=30
)
metric = load_metric("accuracy")

trainer = Trainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()

finetuned_finbert = pipeline(
    model=model, tokenizer=tokenizer, task="sentiment-analysis", return_all_scores=True
)

finetuned_finbert.save_pretrained("nethate_model")
