def method_0:
  
def method_1:
  
def method_2:

def method_3:

def method_4:
  
(x_train, y_train), (x_test, y_test) =  keras.datasets.imdb.load_data(num_words=None, maxlen=None)
words_index = keras.datasets.imdb.get_word_index() #Gets word index where indexes are words
inverted_words_index = dict((i, words_index) for (words_index, i) in words_index.items()) #Translates word_index so that indexes are numbers
line = " ".join(inverted_words_index[i] for i in x_train[0]) #decodes lx_train ine
print(line)
print(y_train[0]) #displays sentiment 0 or 1
  
