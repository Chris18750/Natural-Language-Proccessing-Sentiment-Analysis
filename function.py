# import all required libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pylab as plt

#Resources
#https://docs.w3cub.com/scikit_learn/modules/generated/sklearn.naive_bayes.multinomialnb
#https://www.analyticsvidhya.com/blog/2021/07/performing-sentiment-analysis-with-naive-bayes-classifier/
def MultinomialNB():
  from sklearn.naive_bayes import MultinomialNB
  from sklearn.feature_extraction.text import CountVectorizer
  from sklearn import metrics

  #Load data
  (x_train, y_train), (x_test, y_test) =  keras.datasets.imdb.load_data()

  #Format data with maximum length
  x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=256, padding='post', truncating='post', value=0)
  x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=256, padding='post', truncating='post', value=0)

  model = MultinomialNB()
  
  clf = model.fit(x_train, y_train)
  
  predicted= clf.predict(x_test)

  accuracy = metrics.accuracy_score(y_test, predicted)
  #print("MultinomialNB Accuracy:", accuracy)
  return accuracy
  
#CNN
#Resources:
#https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/
#https://builtin.com/data-science/how-build-neural-network-keras
def method_1():
  #Loading IMDB dataset from keras
  (x_train, y_train), (x_test, y_test) =  keras.datasets.imdb.load_data(num_words=10000)

  #Merging train and test data/class labels to custom split
  data = np.concatenate((x_train, x_test), axis=0)
  targets = np.concatenate((y_train, y_test), axis=0)

  #Training set gets first 20000 examples
  x_train = data[:20000]
  y_train = targets[:20000]
  #Validation set gets next 10000 examples
  x_val = data[20000:30000]
  y_val = targets[20000:30000]
  #Testing gets last 20000 examples
  x_test = data[30000:]
  y_test = targets[30000:]

  #Standardizing review lengths by padding to maxlen 3000
  x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=3000, padding='post', truncating='post', value=0)
  x_val = tf.keras.preprocessing.sequence.pad_sequences(x_val, maxlen=3000, padding='post', truncating='post', value=0)
  x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=3000, padding='post', truncating='post', value=0)

  #Actual model setup
  model = keras.Sequential([
                            keras.layers.Embedding(input_dim=10000, output_dim=8, input_length=3000),
                            keras.layers.Conv1D(filters=5, kernel_size=3, activation='relu'),
                            keras.layers.MaxPooling1D(pool_size=2),
                            keras.layers.Flatten(),
                            keras.layers.Dense(8, activation='relu'),
                            keras.layers.Dropout(0.5),
                            keras.layers.Dense(4, activation='relu'),
                            keras.layers.Dropout(0.5),
                            keras.layers.Dense(1, activation='sigmoid')
  ])
  
  #Training tweaking
  earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3, restore_best_weights=True)

  #Model compilation
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  #Model fitting
  history = model.fit(x_train, y_train, epochs=10, batch_size=50, validation_data=(x_val, y_val), callbacks=[earlystop])

  #Model evaluation
  model.evaluate(x_test, y_test)

#GuassianNB
def method_2():

def method_3():

def method_4():
  
(x_train, y_train), (x_test, y_test) =  keras.datasets.imdb.load_data(num_words=None, maxlen=None)
words_index = keras.datasets.imdb.get_word_index() #Gets word index where indexes are words
inverted_words_index = dict((i, words_index) for (words_index, i) in words_index.items()) #Translates word_index so that indexes are numbers
line = " ".join(inverted_words_index[i] for i in x_train[0]) #decodes lx_train ine
print(line)
print(y_train[0]) #displays sentiment 0 or 1
  
