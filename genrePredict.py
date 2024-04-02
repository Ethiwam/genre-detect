# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 2024

@author: Ethan Iwama
"""
# For word processing
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# For PostgreSQL and the Neural Network
import psycopg2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# Disabling Tensorflow Warnings
import tensorflow as tf
tf.autograph.set_verbosity(0)

#establishing the connection
conn = psycopg2.connect(
   database='pagila', user='postgres', password='------', host='127.0.0.1', port='5432'
)
cursor = conn.cursor()

# Initialize the numpy label array (we'll copy a normal array to the feature array)
cursor.execute("select count(*) from film where film_id < 101") # There are up to 1000 vals; set to 1001 for all values
count = cursor.fetchall()
y = np.empty(count[0][0]) # Labels
titles = []

# Fetching data
cursor.execute("select title, fulltext from film where film_id < 101")
data = cursor.fetchall()

for i, row in enumerate(data):
    for col in row:
        if ('\'' in col):
            if 'drama' in col:
                y[i] = 1
            else:
                y[i] = 0
        else:
            # Tokenizing, Removing Punctuation, To Lowercase
            title = re.sub(r'[^\w\s]', '', col.lower())
            tokens = word_tokenize(title)

            # Removing Stop Words
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
            
            # Stemming
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(word) for word in tokens]

            titles.append(' '.join(tokens))
            
# Vectorization (using TF-IDF)
print('Data Processed, Training Model...')
vectorizer = TfidfVectorizer(min_df=1)
vectorized_titles = vectorizer.fit_transform(titles)

X = vectorized_titles.toarray()

#Closing the connection
conn.close()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Define Keras model
input_size = X_train[0].size # Amount of tokens changes depending on the number of inputs, so we'll scale
model = Sequential()
model.add(Dense(input_size/2, input_shape=(input_size,), activation='softmax'))
model.add(Dense(input_size/4, activation='softmax'))
model.add(Dense(input_size/8, activation='softmax'))
model.add(Dense(1, activation='sigmoid'))

# Compile Keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=2000, batch_size=10, verbose=0)

_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))

# Predicting with the model
print('Testing Model...')
predictions = model.predict(X_test)
predictions = [round(x[0]*10) for x in predictions]

#"""
for i in range(len(X_test)):
    #print('%s => %.5f (expected %d)' % (X_test[i].tolist(), predictions[i], y[i]))
    print('X_test[%d] => %d (expected %d)' % (i, predictions[i], y[i])) 
#"""
