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

#establishing the connection
conn = psycopg2.connect(
   database='pagila', user='postgres', password='p0s+Gr3*', host='127.0.0.1', port='5432'
)
cursor = conn.cursor()

# Initialize the numpy label array (we'll copy a normal array to the feature array)
cursor.execute("select count(*) from film")
count = cursor.fetchall()
y = np.empty(count[0][0]) # Labels
titles = []

# Fetching data
cursor.execute("select title, fulltext from film")
data = cursor.fetchall()

for i, row in enumerate(data):
    for col in row:
        if (len(col) >= 20):
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
vectorizer = TfidfVectorizer()
vectorized_titles = vectorizer.fit_transform(titles)

X = vectorized_titles.toarray()

print('X:')
print(X)
print('Size: %d' % X.size)
print('y:')
print(y)
print('Size: %d' % y.size)

#Closing the connection
conn.close()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
"""
# Define Keras model
model = Sequential()
model.add(Dense(20, input_shape=(1000,), activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile Keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=150, batch_size=10)

_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
"""
