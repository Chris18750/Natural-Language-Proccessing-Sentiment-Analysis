# import all required libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import re
import string
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB, GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#NBs
#Resources:
#https://docs.w3cub.com/scikit_learn/modules/generated/sklearn.naive_bayes.multinomialnb
#https://www.analyticsvidhya.com/blog/2021/07/performing-sentiment-analysis-with-naive-bayes-classifier/
#https://www.learndatasci.com/tutorials/predicting-reddit-news-sentiment-naive-bayes-text-classifiers/
#https://github.com/Ankit152/IMDB-sentiment-analysis/blob/master/imdbSentimentAnalysis.ipynb
def Multinomial(x_train, y_train, x_test, y_test):
  model = MultinomialNB()
  model.fit(x_train, y_train)
  y_pred = model.predict(x_test)
  print("Multinomial train Accuracy: {:.2f}%".format(model.score(x_train, y_train) * 100))
  print("Multinomial test Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
def Gaussian(x_train, y_train, x_test, y_test):
  model = GaussianNB()
  model.fit(x_train, y_train)
  y_pred = model.predict(x_test)
  print("Gaussian train Accuracy: {:.2f}%".format(model.score(x_train, y_train) * 100))
  print("Gaussian test Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
def Complement(x_train, y_train, x_test, y_test):
  model = ComplementNB()
  model.fit(x_train, y_train)
  y_pred = model.predict(x_test)
  print("Complement train Accuracy: {:.2f}%".format(model.score(x_train, y_train) * 100))
  print("Complement test Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
def Bernoulli(x_train, y_train, x_test, y_test):
  model = BernoulliNB()
  model.fit(x_train, y_train)
  y_pred = model.predict(x_test)
  print("Bernoulli train Accuracy: {:.2f}%".format(model.score(x_train, y_train) * 100))
  print("Bernoulli test Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
#CNN
#Resources:
#https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/
#https://builtin.com/data-science/how-build-neural-network-keras
def Convolutional_neural_network():
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

  print("Test:")

  #Model evaluation
  model.evaluate(x_test, y_test)
def clean_text1(text):
  text=text.lower()
  text=re.sub('\[.*?\]','',text)
  text=re.sub('[%s]'%re.escape(string.punctuation),'',text)
  text=re.sub('\w*\d\w*','',text)
  return text

def clean_text2(text):
  text=re.sub('[''"",,,]','',text)
  text=re.sub('\n','',text)
  return text

dataset = pd.read_csv('IMDB Dataset.csv')
dataset['review'] = dataset['review'].str.strip().str.lower()

cleaned1=lambda x:clean_text1(x)
dataset['review']=pd.DataFrame(dataset.review.apply(cleaned1))

cleaned2=lambda x:clean_text2(x)
dataset['review']=pd.DataFrame(dataset.review.apply(cleaned2))

x = dataset.iloc[0:,0].values
y = dataset.iloc[0:,1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 225)

vect = CountVectorizer(max_features=5000, binary=True)
x_train = vect.fit_transform(x_train).toarray()
x_test = vect.transform(x_test).toarray()

Multinomial(x_train, y_train, x_test, y_test)
print()
Gaussian(x_train, y_train, x_test, y_test)
print()
Complement(x_train, y_train, x_test, y_test)
print()
Bernoulli(x_train, y_train, x_test, y_test)
print()
print("Convolutional Neural Network:")
Convolutional_neural_network()
