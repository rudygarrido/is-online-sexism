from collections import Counter

import gensim
import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score


def get_max_lenght(x):
    max_array_size = 0
    tokens = [nltk.word_tokenize(sentence) for sentence in x]
    for i in range(0, len(x)):
        token = tokens[i]
        max_array_size = len(token) if len(token) > max_array_size else max_array_size
    return max_array_size


messages = pd.read_csv('train_all_tasks.csv', encoding='latin-1')
vector_size = get_max_lenght(messages.loc[:, "text"])
messages = messages.drop(labels=["rewire_id", "label_category", "label_vector"], axis=1)
messages.columns = ["text", "label_sexist"]

messages['text_clean'] = messages['text'].apply(lambda x: gensim.utils.simple_preprocess(x))

messages['label_sexist'] = messages['label_sexist'].map({'sexist': 1, 'not sexist': 0})

X_train, X_test, y_train, y_test = train_test_split(messages['text_clean'], messages['label_sexist'], test_size=0.2,
                                                    stratify=messages['label_sexist'])

w2v_model = gensim.models.Word2Vec(X_train,
                                   vector_size=vector_size,
                                   window=5,
                                   min_count=2)

words = set(w2v_model.wv.index_to_key)
X_train_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                         for ls in X_train])

X_test_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                        for ls in X_test])

X_train_vect_avg = []
for v in X_train_vect:
    if v.size:
        X_train_vect_avg.append(v.mean(axis=0))
    else:
        X_train_vect_avg.append(np.zeros(vector_size, dtype=float))

X_test_vect_avg = []
for v in X_test_vect:
    if v.size:
        X_test_vect_avg.append(v.mean(axis=0))
    else:
        X_test_vect_avg.append(np.zeros(vector_size, dtype=float))

rf = RandomForestClassifier(bootstrap=True, random_state=42, n_estimators=1000)
rf_model = rf.fit(X_train_vect_avg, y_train.values.ravel())

y_pred = rf_model.predict(X_test_vect_avg)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('Precision: {} / Recall: {} / Accuracy: {} / F1: {}'.format(
    round(precision, 3), round(recall, 3), round((y_pred == y_test).sum() / len(y_pred), 3), round(f1, 3)))
