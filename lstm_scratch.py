import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from collections import Counter

from pandas import Index
from sklearn.model_selection import train_test_split

torch.manual_seed(1)
random_state_value = 56
test_split = 0.20
val_split = 0.12

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def index_words(x, y, test_sentence, skip_top = 0):
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
    test_sentence_array = []
    for token_word in nltk.word_tokenize(test_sentence):
        test_sentence_array.append(tokens_counted[token_word])
    return x, y, max_array_size, len(tokens_counted), test_sentence_array


df = pd.read_csv('train_all_tasks.csv')
x = df.loc[:, "text"]
y = df.loc[:, "label_category"]
test_sentence = "I hate women so much"
x, y, max_words, tokens_size, test_sentence = index_words(x, y, test_sentence)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, random_state=random_state_value,
                                                    shuffle=True, stratify=y)

#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_split,
#                                                  random_state=random_state_value, shuffle=True, stratify=y_train)
"""
training_data = [
    # Tags are: DET - determiner; NN - noun; V - verb
    # For example, the word "The" is a determiner
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]


training_data = pd.merge(x_train, y_train, left_index=True, right_index=True)
#pd.concat([x_train, y_train], axis=1)
#pd. merge(x_train, y_train, left_index=True, right_index=True)
word_to_ix = {}
# For each words-list (sentence) and tags-list in each tuple of training_data
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:  # word has not been assigned an index yet
            word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index
print(word_to_ix)
"""
tag_to_ix = {"None": 0, "Threats": 1, "Derogation": 2,  "Animosity": 3, "Prejudice": 4}  # Assign each tag with a unique index

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 32
HIDDEN_DIM = 32

print(torch.tensor(test_sentence, dtype=torch.long))

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

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, tokens_size, len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = torch.tensor(test_sentence, dtype=torch.long)
    tag_scores = model(inputs)
    print(tag_scores)

for epoch in range(25):  # again, normally you would NOT do 300 epochs, it is toy data
    for index, sentence in x_train.items():
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = torch.tensor(sentence, dtype=torch.long)
        targets = torch.tensor([y_train.loc[index]] * len(sentence), dtype=torch.long)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

model.eval()
total_acc, total_count = 0, 0
# See what the scores are after training
with torch.no_grad():
    for index, sentence in x_test.items():
        predicted_label = model(torch.tensor(sentence, dtype=torch.long))
        label = torch.tensor([y_test.loc[index]] * len(sentence), dtype=torch.long)
        loss = loss_function(predicted_label, label)
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)

print(total_acc/total_count)

