# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
'''
from __future__ import print_function

import nltk
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from keras.datasets import reuters
from keras import utils as np_utils
from nltk.tokenize import word_tokenize
from collections import Counter
import csv

def equalize_samples (x_positive, x_negative, test_split):
    x_train_positive = x_positive[:int(len(x_positive) * (1 - test_split))]
    x_train_negative = x_negative[:int(len(x_negative) * (1 - test_split))]
    x_train = pd.concat([x_train_positive, x_train_negative])
    y_train = pd.concat([pd.Series(1, index=range(len(x_train_positive))),
                         pd.Series(0, index=range(len(x_train_negative)))])

    x_test_positive = x_positive[int(len(x_positive) * (1 - test_split)):]
    x_test_negative = x_negative[int(len(x_negative) * (1 - test_split)):]
    x_test = pd.concat([x_test_positive, x_test_negative])
    y_test = pd.concat([pd.Series(1, index=range(len(x_test_positive))),
                        pd.Series(0, index=range(len(x_test_negative)))])


    return x_train, y_train, x_test, y_test

def index_words(x, y, skip_top, use_previous = True):
    max_array_size = 0
    x_positive = pd.Series()
    x_negative = pd.Series()
    tokens = [ nltk.word_tokenize(sentence) for sentence in x ]
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
        if y[i] == 'sexist':
            y[i] = 1
            x_positive = pd.concat([x_positive, pd.Series([token_indexed])])
        else:
            y[i] = 0
            x_negative = pd.concat([x_negative,pd.Series([token_indexed])])
    return x_positive, x_negative , max_array_size
    #return x, y  , max_array_size


def word2vec():
    batch_size = 32
    epochs = 5
    skip_top = 40
    test_split = 0.3
    num_test = 5
    sum_accuracy = 0
    with open('results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Test', 'Score', 'Accuracy'])
        for test in range(num_test):
            print('Loading data...')

            df = pd.read_csv('train_all_tasks.csv')

            x = df.loc[:, "text"]
            y = df.loc[:, "label_sexist"]

            #x, y, max_words = index_words(x, y, skip_top)
            x_positive, x_negative, max_words = index_words(x, y, skip_top)
            x_train, y_train, x_test, y_test = equalize_samples(x_positive, x_negative, test_split)

            """
            x_train = x[:int(len(x) * (1 - test_split))]
            y_train = y[:int(len(y) * (1 - test_split))]
            x_test = x[int(len(x) * (1 - test_split)):]
            y_test = y[int(len(y) * (1 - test_split)):]
            """

            """""
            (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words,
                                                                     test_split=test_split, skip_top=skip_top)
            """""
            print(len(x_train), 'train sequences')
            print(len(x_test), 'test sequences')

            num_classes = np.max(y_train) + 1  # 2
            print(num_classes, 'classes')

            print('Vectorizing sequence data...')
            tokenizer = Tokenizer(num_words=max_words)
            x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
            x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
            print('x_train shape:', x_train.shape)
            print('x_test shape:', x_test.shape)

            print('Convert class vector to binary class matrix '
                  '(for use with categorical_crossentropy)')
            y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)
            print('y_train shape:', y_train.shape)
            print('y_test shape:', y_test.shape)

            print('Building model...')
            model = Sequential()
            model.add(Dense(512, input_shape=(max_words,)))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(num_classes))
            model.add(Activation('softmax'))

            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            model.fit(x_train, y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=1,
                                validation_split=0.1)
            score = model.evaluate(x_test, y_test,
                                   batch_size=batch_size, verbose=1)

            writer.writerow([test, score[0], score[1]])
            print('Test number ', test)
            print('Test score:', score[0])
            print('Test accuracy:', score[1])
            sum_accuracy += score[1]
        print('Mean accuracy: ', sum_accuracy / num_test)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    word2vec()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
