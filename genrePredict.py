# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 2024

@author: Ethan Iwama
"""
import psycopg2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#establishing the connection
conn = psycopg2.connect(
   database='pagila', user='postgres', password='p0s+Gr3*', host='127.0.0.1', port='5432'
)
#Creating a cursor object using the cursor() method
cursor = conn.cursor()

#Executing a PostgreSQL function using the execute() method
cursor.execute("select title from film where film_id < 11")

# Fetch data
data = cursor.fetchall()
for title in data:
    title = str(title)
    print(title.replace('(', '').replace('\'', '').replace(',', '').replace(')', ''))

#Closing the connection
conn.close()

# Load data
#-----

# Split data
#-----
"""
# Define Keras model
model = Sequential()
model.add(Dense(20, input_shape=(8,), activation='relu'))
# Add more model layers

# Compile Keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=150, batch_size=10)
"""