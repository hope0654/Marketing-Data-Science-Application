# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 09:48:55 2021

@author: houpeng
"""

from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
rus = RandomUnderSampler(sampling_strategy='majority')
train_X,train_y = rus.fit_resample(train_X,train_y) 

##-----------oversampling---------------
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
ros = RandomOverSampler(sampling_strategy='minority')
train_X,train_y = ros.fit_resample(train_X,train_y) 

## import embedding file
EMBEDDING_FILE = '../embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
embeddings_index = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

## some config values 
embed_size = 300 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a question to use

train_data, val_data = train_test_split(train, test_size = 0.2, random_state = 42)

## fill up the missing values
train_X = train_data["question_text"].fillna("_na_").values
val_X = val_data["question_text"].fillna("_na_").values
test_X = test_df["question_text"].fillna("_na_").values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

## Get the target values
train_y = train_data['target'].values
val_y = val_data['target'].values

## word embedding
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = (np.random.rand(nb_words, embed_size) - 0.5) / 5.0
for word, i in word_index.items():
    if i >= max_features: continue
    if word in embeddings_index:
        embedding_vector = embeddings_index.get_vector(word)
        embedding_matrix[i] = embedding_vector

##----------------logistic regression------------------
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(train_X, train_y)
y_pred = classifier.predict(test_X)
y_pred_proba = classifier.predict_proba(test_X)

## set up GRU model
def get_model():
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(GRU(64, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(64, activation="relu")(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(1, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = get_model()
print(model.summary())

## train GRU model
history = model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
y_test = model.predict(test_X)
y_test = y_test.reshape((-1, 1))

## predict 
kaggle = test_df.copy()
pred_test_y = (y_test>0.34506).astype(int)
kaggle['prediction'] = pred_test_y
original_test_y = test_df['target']

## import package
from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score,confusion_matrix

## calculate
accuracy = accuracy_score(original_test_y, y_pred)
f1 = f1_score(original_test_y, y_pred)
precision = precision_score(original_test_y, y_pred)
recall = recall_score(original_test_y, y_pred)
cm = confusion_matrix(original_test_y, y_pred)

print('accuracy:', accuracy)
print('F1 score:', f1)
print('precision score:',precision)
print('recall score:', recall)
print('confusion matrix:', cm)
