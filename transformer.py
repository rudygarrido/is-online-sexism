import torch
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, \
    TrainingArguments, Trainer
import pandas as pd
import evaluate
import numpy as np
from sklearn.model_selection import train_test_split

id2label = {0: "none", 1: "1.1 threats of harm", 2: "1.2 incitement and encouragement of harm", 3: "2.1 descriptive "
                                                                                                   "attacks",
            4: "2.2 aggressive and emotive attacks", 5: "2.3 dehumanising attacks & overt sexual objectification",
            6: "3.1 casual use of gendered slurs, profanities, and insults", 7: "3.2 immutable gender differences and "
                                                                                "gender stereotypes",
            8: "3.3 backhanded gendered compliments", 9: "3.4 condescending explanations or unwelcome advice",
            10: "4.1 supporting mistreatment of individual women", 11: "4.2 supporting systemic discrimination "
                                                                       "against women as a group"}
label2id = {"none": 0, "1.1 threats of harm": 1, "1.2 incitement and encouragement of harm": 2, "2.1 descriptive "
                                                                                                "attacks": 3,
            "2.2 aggressive and emotive attacks": 4, "2.3 dehumanising attacks & overt sexual objectification": 5,
            "3.1 casual use of gendered slurs, profanities, and insults": 6, "3.2 immutable gender differences and "
                                                                             "gender stereotypes": 7, "3.3 backhanded "
                                                                                                      "gendered "
                                                                                                      "compliments":
                8, "3.4 condescending explanations or unwelcome advice": 9, "4.1 supporting mistreatment of "
                                                                            "individual women": 10, "4.2 supporting "
                                                                                                    "systemic "
                                                                                                    "discrimination "
                                                                                                    "against women as "
                                                                                                    "a group": 11}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def preprocess_function(text):
    '''for index in data.index:
        tokenized = tokenizer(data.iloc[index]["text"], truncation=True)
        data.loc[index, "text"] = tokenized
        #data[index] = data[index].apply(lambda x: tokenizer(data.iloc[index]["text"], truncation=True))
    return data
    '''
    return tokenizer(text, truncation=True)


accuracy = evaluate.load("accuracy")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
dataset = pd.read_csv('train_all_tasks.csv')
dataset.drop(labels=["rewire_id", "label_category", "label_sexist"], axis=1, inplace=True)
dataset.rename(columns={"label_vector": "label"}, inplace=True)
print(dataset[0])
dataset['label'] = dataset['label'].map(
    {"none": 0, "1.1 threats of harm": 1, "1.2 incitement and encouragement of harm": 2, "2.1 descriptive "
                                                                                         "attacks": 3,
     "2.2 aggressive and emotive attacks": 4, "2.3 dehumanising attacks & overt sexual objectification": 5,
     "3.1 casual use of gendered slurs, profanities, and insults": 6, "3.2 immutable gender differences and "
                                                                      "gender stereotypes": 7, "3.3 backhanded "
                                                                                               "gendered "
                                                                                               "compliments":
         8, "3.4 condescending explanations or unwelcome advice": 9, "4.1 supporting mistreatment of "
                                                                     "individual women": 10, "4.2 supporting "
                                                                                             "systemic "
                                                                                             "discrimination "
                                                                                             "against women as "
                                                                                             "a group": 11})
#dataset = preprocess_function(dataset)
dataset["text"] = dataset["text"].apply(preprocess_function)
train, test = train_test_split(dataset, test_size=0.2)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=12, id2label=id2label, label2id=label2id
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
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

inputs = tokenizer("Black women have too much privilege, they do not need any reparations", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    print(model.config.id2label[predicted_class_id])
