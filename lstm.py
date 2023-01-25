from collections import Counter

import pandas as pd
import torch
import torchtext.vocab
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from torch.utils.data import DataLoader
import torch.nn.functional as F
import nltk
import itertools as IT

device = 'cpu'

torch.manual_seed(1)

train_test_ratio = 0.10
train_valid_ratio = 0.80

df_raw = pd.read_csv("train_all_tasks.csv")

df_raw['label_sexist'] = (df_raw['label_sexist'] == 'sexist').astype('int')

df_raw = df_raw.reindex(columns=['label_sexist', 'text'])
text_raw = df_raw.loc[:, "text"]
tokens = [nltk.word_tokenize(sentence) for sentence in text_raw]
tokens_counted = Counter([item for sublist in tokens for item in sublist])
longest_array = int(df_raw["text"].str.len().max())

df_sexist = df_raw[df_raw['label_sexist'] == 1]
df_not_sexist = df_raw[df_raw['label_sexist'] == 0]

# Train-test split
df_sexist_full_train, df_sexist_test = train_test_split(df_sexist, train_size=train_test_ratio, random_state=1)
df_not_sexist_full_train, df_not_sexist_test = train_test_split(df_not_sexist, train_size=train_test_ratio,
                                                                random_state=1)

# Train-valid split
df_sexist_train, df_sexist_valid = train_test_split(df_sexist_full_train, train_size=train_valid_ratio, random_state=1)
df_not_sexist_train, df_not_sexist_valid = train_test_split(df_not_sexist_full_train, train_size=train_valid_ratio,
                                                            random_state=1)

# Concatenate splits of different labels
df_train = pd.concat([df_sexist_train, df_not_sexist_train], ignore_index=True, sort=False)
df_valid = pd.concat([df_sexist_valid, df_not_sexist_valid], ignore_index=True, sort=False)
df_test = pd.concat([df_sexist_test, df_not_sexist_test], ignore_index=True, sort=False)

# Write preprocessed data
df_train.to_csv('lstm_data/train.csv', index=False)
df_valid.to_csv('lstm_data/valid.csv', index=False)
df_test.to_csv('lstm_data/test.csv', index=False)


class CustomDataset(Dataset):
    def __init__(self, type):
        if type == 'train':
            self.data = df_train
        if type == 'validate':
            self.data = df_valid
        if type == 'test':
            self.data = df_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        text = item["text"]
        token_indexed = []
        for token_word in nltk.word_tokenize(text):
            token_indexed.append(tokens_counted[token_word])
        if len(token_indexed) < longest_array:
            token_indexed += [0] * (longest_array - len(token_indexed))
        # return {'text': token_indexed, 'label_field': [item["label_sexist"]]}
        returned = torch.tensor(token_indexed), item["label_sexist"]
        return returned


train = CustomDataset(type='train')
test = CustomDataset(type='test')
valid = CustomDataset(type='validate')

train_dataloader = DataLoader(train, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(valid, batch_size=32, shuffle=True)

train_iter = iter(train_dataloader)
valid_iter = iter(valid_dataloader)
test_iter = iter(test_dataloader)

"""
# Fields

#text_field = data.Field(tokenize='spacy', lower=True, include_lengths=True, batch_first=True)
#label_field = data.LabelField(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)

label_field = torchtext.data.Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
text_field = torchtext.data.Field(tokenize='spacy', lower=True, include_lengths=True, batch_first=True)
fields = [('label_sexist', label_field), ('text', text_field)]

# TabularDataset

train, valid, test = TabularDataset.splits(path='lstm_data/', train='train.csv',
                                           validation='valid.csv', test='test.csv',
                                           format='CSV', fields=fields, skip_header=True)
#train, valid, test = data.split

# Iterators

train_iter = BucketIterator(train, batch_size=32, sort_key=lambda x: len(x.text),
                            device=device, sort=True, sort_within_batch=True)
valid_iter = BucketIterator(valid, batch_size=32, sort_key=lambda x: len(x.text),
                            device=device, sort=True, sort_within_batch=True)
test_iter = BucketIterator(test, batch_size=32, sort_key=lambda x: len(x.text),
                           device=device, sort=True, sort_within_batch=True)
                           
# Vocabulary

text_field.build_vocab(train, min_freq=3)


"""


# vocab_train = torchtext.vocab.build_vocab_from_iterator(train_iter)

class LSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTM, self).__init__()
        """
        self.embedding = nn.Embedding(len(tokens_counted), 300)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(2 * dimension, 1)
        """
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        """
            text_emb = self.embedding(text)

            packed_input = pack_padded_sequence(text_emb, longest_array, batch_first=True, enforce_sorted=False)
            packed_output, _ = self.lstm(packed_input)
            output, _ = pad_packed_sequence(packed_output, batch_first=True)

            out_forward = output[range(len(output)), longest_array - 1, :self.dimension]
            out_reverse = output[:, 0, self.dimension:]
            out_reduced = torch.cat((out_forward, out_reverse), 1)
            text_fea = self.drop(out_reduced)

            text_fea = self.fc(text_fea)
            text_fea = torch.squeeze(text_fea, 1)
            text_out = torch.sigmoid(text_fea)

            return text_out
            """
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        #embeds 32*250*250
        #ltmout 32*250*64
        #h2t 64*32
        #lstm_view = lstm_out.view(len(sentence), self.hidden_dim)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


# Save and Load Functions

def save_checkpoint(save_path, model, optimizer, valid_loss):
    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, optimizer):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path == None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


def train(model,
          optimizer,
          criterion=nn.BCELoss(),
          train_loader=train_iter,
          valid_loader=valid_iter,
          num_epochs=5,
          eval_every=len(train_iter) // 2,
          file_path='lstm_data/',
          best_valid_loss=float("Inf")):
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    # model.train()
    for epoch in range(num_epochs):
        for item in train_loader:
            targets = item[1].to(device)
            texts = item[0].to(device)
            # text_len = len(text).to(device)
            for text, target in zip(texts, targets):
                model.zero_grad()
                tag_scores = model(text)

                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                target_tensor = torch.tensor([target]*EMBEDDING_DIM, dtype=torch.int64)
                loss = loss_function(tag_scores, target_tensor)
                loss.backward()
                optimizer.step()
                """
                output = model(text)
    
                loss = criterion(output, item['label_field'])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                # update running values
                running_loss += loss.item()
                global_step += 1
    
                # evaluation step
                if global_step % eval_every == 0:
                    model.eval()
                    with torch.no_grad():
                        # validation loop
                        for (labels, (text, text_len)), _ in valid_loader:
                            labels = labels.to(device)
                            text = text.to(device)
                            text_len = text_len.to(device)
                            output = model(text, text_len)
    
                            loss = criterion(output, labels)
                            valid_running_loss += loss.item()
    
                    # evaluation
                    average_train_loss = running_loss / eval_every
                    average_valid_loss = valid_running_loss / len(valid_loader)
                    train_loss_list.append(average_train_loss)
                    valid_loss_list.append(average_valid_loss)
                    global_steps_list.append(global_step)
    
                    # resetting running values
                    running_loss = 0.0
                    valid_running_loss = 0.0
                    model.train()
    
                    # print progress
                    print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                          .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                                  average_train_loss, average_valid_loss))
    
                    # checkpoint
                    if best_valid_loss > average_valid_loss:
                        best_valid_loss = average_valid_loss
                        save_checkpoint(file_path + '/model.pt', model, optimizer, best_valid_loss)
                        save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
                """
    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')


# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = longest_array
HIDDEN_DIM = 64
tag_to_ix = {"sexist": 0, "not_sexist": 1}

model = LSTM(EMBEDDING_DIM, HIDDEN_DIM, len(tokens_counted), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
train(model=model, optimizer=optimizer, num_epochs=10)
"""
model = LSTM().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
"""

train_loss_list, valid_loss_list, global_steps_list = load_metrics('lstm_data/metrics.pt')
plt.plot(global_steps_list, train_loss_list, label='Train')
plt.plot(global_steps_list, valid_loss_list, label='Valid')
plt.xlabel('Global Steps')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Evaluation Function

def evaluate(model, test_loader, version='text', threshold=0.5):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for (labels, (text, text_len)), _ in test_loader:
            labels = labels.to(device)
            text = text.to(device)
            text_len = text_len.to(device)
            output = model(text, text_len)

            output = (output > threshold).int()
            y_pred.extend(output.tolist())
            y_true.extend(labels.tolist())

    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1, 0], digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['Sexist', 'Not Sexist'])
    ax.yaxis.set_ticklabels(['Sexist', 'Not Sexist'])


best_model = LSTM().to(device)
optimizer = optim.Adam(best_model.parameters(), lr=0.001)

load_checkpoint('lstm_data/model.pt', best_model, optimizer)
evaluate(best_model, test_iter)
