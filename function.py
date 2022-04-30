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
  
  
def method_1():
  
def method_2():

def method_3():

def method_4():
  
(x_train, y_train), (x_test, y_test) =  keras.datasets.imdb.load_data(num_words=None, maxlen=None)
words_index = keras.datasets.imdb.get_word_index() #Gets word index where indexes are words
inverted_words_index = dict((i, words_index) for (words_index, i) in words_index.items()) #Translates word_index so that indexes are numbers
line = " ".join(inverted_words_index[i] for i in x_train[0]) #decodes lx_train ine
print(line)
print(y_train[0]) #displays sentiment 0 or 1
  
