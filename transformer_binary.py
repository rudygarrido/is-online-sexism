import torch
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, \
    TrainingArguments, Trainer, AutoConfig
import pandas as pd
import evaluate
import numpy as np
from sklearn.model_selection import train_test_split

id2label = {0: "not sexist", 1: "sexist"}
label2id = {'sexist': 1, 'not sexist': 0}


class SexismDatasset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    results = {}
    results.update(f1_metric.compute(predictions=predictions, references=labels))
    results.update(recall_metric.compute(predictions=predictions, references=labels))
    results.update(precision_metric.compute(predictions=predictions, references=labels))
    results.update(accuracy_metric.compute(predictions=predictions, references=labels))

    return results


f1_metric = evaluate.load("f1")
recall_metric = evaluate.load("recall")
precision_metric = evaluate.load("precision")
accuracy_metric = evaluate.load("accuracy")
config = AutoConfig.from_pretrained("distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", config=config)
dataset = pd.read_csv('train_all_tasks.csv')
dataset.drop(labels=["rewire_id", "label_category", "label_vector"], axis=1, inplace=True)
dataset.rename(columns={"label_sexist": "label"}, inplace=True)
dataset['label'] = dataset['label'].map(
    {'sexist': 1, 'not sexist': 0})
train, test = train_test_split(dataset, test_size=0.2, shuffle=True, stratify=dataset['label'])
train_texts, train_labels = train["text"].tolist(), train["label"].tolist()
test_texts, test_labels = test["text"].tolist(), test["label"].tolist()


train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

train_dataset = SexismDatasset(train_encodings, train_labels)
test_dataset = SexismDatasset(test_encodings, test_labels)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="is_online_sexism_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

best_trial = trainer.hyperparameter_search(
    direction="maximize",
    backend="ray",
    n_trials=2
)

print(best_trial)