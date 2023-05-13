import os
from functools import partial

import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from collections import Counter
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from pandas import Index
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix

torch.manual_seed(1)
random_state_value = 86
test_split = 0.30
val_split = 0.12
tag_to_ix = {"None": 0, "Threats": 1, "Derogation": 2, "Animosity": 3,
             "Prejudice": 4}  # Assign each tag with a unique index
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
    """test_sentence_array = []
    for token_word in nltk.word_tokenize(test_sentence):
        test_sentence_array.append(tokens_counted[token_word])"""
    return x, y, max_array_size, len(tokens_counted) #, test_sentence_array



df = pd.read_csv('train_all_tasks.csv')
x = df.loc[:, "text"]
y = df.loc[:, "label_category"]
test_sentence = "I hate women so much"
x, y, max_words, tokens_size = index_words(x, y, 10)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, random_state=random_state_value,
                                                        shuffle=True)

    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, random_state=random_state_value, shuffle=True, stratify=y)
    #return x_train, y_train, x_test, y_test, tokens_size



# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.


#print(torch.tensor(test_sentence, dtype=torch.long))

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence_in):
        embeds = self.word_embeddings(sentence_in)
        lstm_out, _ = self.lstm(embeds.view(len(sentence_in), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence_in), -1))
        tag_scores_forward = F.log_softmax(tag_space, dim=1)
        return tag_scores_forward

def train(config, num_epochs = 10, checkpoint_dir=None):

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

    for epoch in range(num_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
        running_loss = 0.0
        epoch_steps = 0
        for index, sentence in x_train.items():
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence = torch.tensor(sentence, dtype=torch.long)
            targets = torch.tensor([y_train.loc[index]] * len(sentence), dtype=torch.long)

            # Step 3. Run our forward pass.
            tag_scores = model(sentence)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if index % 2000 == 1999:  # print every 2000 mini-batches
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
                    precision=(precision/len(x_test)), recall=(recall/len(x_test)), f1=(f1/len(x_test)))
    print("Finished Training")



# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
"""with torch.no_grad():
    inputs = torch.tensor(test_sentence, dtype=torch.long)
    tag_scores = model(inputs)
    print(tag_scores)"""

def test_metrics(model, device="cpu"):
    #model.eval()
    total_acc, total_count, precision, recall, f1 = 0, 0, 0, 0, 0
    # See what the scores are after training
    with torch.no_grad():
        for index, sentence in x_test.items():
            predicted_label = model(torch.tensor(sentence, dtype=torch.long))
            label = torch.tensor([y_test.loc[index]] * len(sentence), dtype=torch.long)
            prediction, inds = torch.max(predicted_label, dim=1)
            prediction[prediction < 0] = 0

            if (torch.count_nonzero(prediction) > 0):
                print("NON ZERO")

            # cm = confusion_matrix(label, prediction.type(torch.int64))

            precision += precision_metric(prediction.type(torch.int64), label)


            recall += recall_metric(prediction.type(torch.int64), label)


            f1 += f1_metric(prediction.type(torch.int64), label)

            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)

    return total_acc / total_count, (precision / len(x_test)).item(), (recall / len(x_test)).item(), (f1 / len(x_test)).item()

def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2, checkpoint_dir=None):
    config = {
        #"lr": tune.choice([0.01, 0.001, 0.0001]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16]),
        "hidden_dim": tune.choice([8, 16, 32, 64]),
        "embedding_dim": tune.choice([8, 16, 32, 64]),
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # ``parameter_columns=["lr", "batch_size", "hidden_dim", "embedding_dim"]``,
        metric_columns=["loss", "accuracy", "precision", "recall", "f1", "training_iteration"
    ])
    result = tune.run(
        partial(train, num_epochs=max_num_epochs),
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

    best_trained_model = LSTMTagger(best_trial.config["embedding_dim"], best_trial.config["hidden_dim"], tokens_size, len(tag_to_ix))
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

main(max_num_epochs=3, gpus_per_trial=0)

"""
#model.eval()
total_acc, total_count, precision, recall, f1 = 0, 0, 0, 0, 0
# See what the scores are after training
with torch.no_grad():
    for index, sentence in x_test.items():
        predicted_label = model(torch.tensor(sentence, dtype=torch.long))
        label = torch.tensor([y_test.loc[index]] * len(sentence), dtype=torch.long)
        loss = loss_function(predicted_label, label)
        prediction, inds = torch.max(predicted_label, dim=1)
        prediction[prediction < 0] = 0

        if (torch.count_nonzero(prediction) > 0):
            print("NON ZERO")

        # cm = confusion_matrix(label, prediction.type(torch.int64))
        precision_metric = MulticlassPrecision(num_classes=5)
        precision += precision_metric(prediction.type(torch.int64), label)

        recall_metric = MulticlassRecall(num_classes=5)
        recall += recall_metric(prediction.type(torch.int64), label)

        f1_metric = MulticlassF1Score(num_classes=5)
        f1 += f1_metric(prediction.type(torch.int64), label)

        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)

print(total_acc / total_count)
print(precision / len(x_test))
print(recall / len(x_test))
print(f1 / len(x_test))
"""