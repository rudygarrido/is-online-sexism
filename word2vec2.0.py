# Read in the data and clean up column names
import gensim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

pd.set_option('display.max_colwidth', 100)

messages = pd.read_csv('train_all_tasks.csv', encoding='latin-1')
messages = messages.drop(labels = ["rewire_id", "label_category", "label_vector"], axis = 1)
messages.columns = ["text", "label_sexist"]

messages['text_clean'] = messages['text'].apply(lambda x: gensim.utils.simple_preprocess(x))

# Encoding the label column
messages['label_sexist']=messages['label_sexist'].map({'sexist':1,'not sexist':0})

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split (messages['text_clean'], messages['label_sexist'] , test_size=0.2)

# Train the word2vec model
w2v_model = gensim.models.Word2Vec(X_train,
                                   vector_size=100,
                                   window=5,
                                   min_count=2)

words = set(w2v_model.wv.index_to_key )
X_train_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                         for ls in X_train])

X_test_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                         for ls in X_test])

# Compute sentence vectors by averaging the word vectors for the words contained in the sentence
X_train_vect_avg = []
for v in X_train_vect:
    if v.size:
        X_train_vect_avg.append(v.mean(axis=0))
    else:
        X_train_vect_avg.append(np.zeros(100, dtype=float))

X_test_vect_avg = []
for v in X_test_vect:
    if v.size:
        X_test_vect_avg.append(v.mean(axis=0))
    else:
        X_test_vect_avg.append(np.zeros(100, dtype=float))

rf = RandomForestClassifier()
rf_model = rf.fit(X_train_vect_avg, y_train.values.ravel())

y_pred = rf_model.predict(X_test_vect_avg)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('Precision: {} / Recall: {} / Accuracy: {} / F1: {}'.format(
    round(precision, 3), round(recall, 3), round((y_pred==y_test).sum()/len(y_pred), 3), round(f1,3)))