import os
from collections import Counter
from functools import partial

import nltk
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import train_test_split
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score

torch.manual_seed(1)
random_state_value = 86
test_split = 0.30
val_split = 0.12
tag_to_ix = {"None": 0, "Threats": 1, "Derogation": 2, "Animosity": 3,
             "Prejudice": 4}
precision_metric = MulticlassPrecision(num_classes=5)
recall_metric = MulticlassRecall(num_classes=5)
f1_metric = MulticlassF1Score(num_classes=5)


def index_words(x, y, skip_top=0):
    max_array_size = 0
    tokens = [nltk.word_tokenize(sentence) for sentence in x]
    tokens_counted = Counter([item for sublist in tokens for item in sublist])
    if skip_top > 0:
        skip_tokens = []
        top_tokens = tokens_counted.most_common()[0: skip_top]
        for token in top_tokens:
            skip_tokens.append(token[0])
    for i in range(0, len(x)):
        token = tokens[i]
        max_array_size = len(token) if len(token) > max_array_size else max_array_size
        token_indexed = []
        for token_word in token:
            if skip_top > 0 and token_word in skip_tokens:
                continue
            token_indexed.append(tokens_counted[token_word])
        x[i] = token_indexed
        if y[i] == 'none':
            y[i] = 0
        else:
            y[i] = int(y[i][0:1])
    return x, y, max_array_size, len(tokens_counted)


df = pd.read_csv('train_all_tasks.csv')
x = df.loc[:, "text"]
y = df.loc[:, "label_category"]
test_sentence = "I hate women so much"
x, y, max_words, tokens_size = index_words(x, y, 10)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, random_state=random_state_value,
                                                    shuffle=True)


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence_in):
        embeds = self.word_embeddings(sentence_in)
        lstm_out, _ = self.lstm(embeds.view(len(sentence_in), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence_in), -1))
        tag_scores_forward = F.log_softmax(tag_space, dim=1)
        return tag_scores_forward


def train(config, checkpoint_dir=None):
    model = LSTMTagger(config["embedding_dim"], config["hidden_dim"], tokens_size, len(tag_to_ix))

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    for epoch in range(config["num_epochs"]):
        running_loss = 0.0
        epoch_steps = 0
        for index, sentence in x_train.items():
            model.zero_grad()
            sentence = torch.tensor(sentence, dtype=torch.long)
            targets = torch.tensor([y_train.loc[index]] * len(sentence), dtype=torch.long)
            tag_scores = model(sentence)
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_steps += 1
            if index % 2000 == 1999:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, index + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        val_loss = 0.0
        val_steps = 0
        total = 0
        correct, precision, recall, f1 = 0, 0, 0, 0
        for index, sentence in x_test.items():
            with torch.no_grad():
                outputs = model(torch.tensor(sentence, dtype=torch.long))
                label = torch.tensor([y_test.loc[index]] * len(sentence), dtype=torch.long)
                prediction, inds = torch.max(outputs, dim=1)
                prediction[prediction < 0] = 0

                total += label.size(0)
                correct += (prediction == label).sum().item()

                precision += precision_metric(prediction.type(torch.int64), label)
                recall += recall_metric(prediction.type(torch.int64), label)
                f1 += f1_metric(prediction.type(torch.int64), label)

                loss = loss_function(outputs, label)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total,
                    precision=(precision / len(x_test)), recall=(recall / len(x_test)), f1=(f1 / len(x_test)))
    print("Finished Training")


def test_metrics(model, device="cpu"):
    total_acc, total_count, precision, recall, f1 = 0, 0, 0, 0, 0

    with torch.no_grad():
        for index, sentence in x_test.items():
            predicted_label = model(torch.tensor(sentence, dtype=torch.long))
            label = torch.tensor([y_test.loc[index]] * len(sentence), dtype=torch.long)
            prediction, inds = torch.max(predicted_label, dim=1)
            prediction[prediction < 0] = 0

            precision += precision_metric(prediction.type(torch.int64), label)

            recall += recall_metric(prediction.type(torch.int64), label)

            f1 += f1_metric(prediction.type(torch.int64), label)

            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)

    return total_acc / total_count, (precision / len(x_test)).item(), (recall / len(x_test)).item(), (
            f1 / len(x_test)).item()


def main(num_samples=10, gpus_per_trial=2, checkpoint_dir=None):
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "hidden_dim": tune.choice([8, 16, 32, 64]),
        "embedding_dim": tune.choice([8, 16, 32, 64]),
        "num_epochs": tune.choice([3, 4, 5, 6, 7, 8])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "precision", "recall", "f1", "training_iteration"
                        ])
    result = tune.run(
        partial(train),
        resources_per_trial={"cpu": 12, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = LSTMTagger(best_trial.config["embedding_dim"], best_trial.config["hidden_dim"], tokens_size,
                                    len(tag_to_ix))
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    if checkpoint_dir:
        best_checkpoint_dir = best_trial.checkpoint.value
        model_state, optimizer_state = torch.load(os.path.join(
            best_checkpoint_dir, "checkpoint"))
        best_trained_model.load_state_dict(model_state)

    accuracy, precission, recall, f1 = test_metrics(best_trained_model, device)
    print("Best trial test set accuracy: {}, precission: {}, recall: {}, f1: {}"
          .format(accuracy, precission, recall, f1))


main(gpus_per_trial=0)
