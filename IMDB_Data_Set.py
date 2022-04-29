#Stores IMDB data set and defines functions to get the data
class IMDB_Data_Set:
  def __init__(self):
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=None, maxlen=None) #Gets Reuters data set
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test
    self.words_index = keras.datasets.imdb.get_word_index() #Gets word index where indexes are words
    self.inverted_words_index = dict((i, words_index) for (words_index, i) in self.words_index.items()) #Translates word_index so that indexes are numbers
  
  #Gets specified line in x_train where data are numbers
  def get_raw_x_train_line(self, line_number):
    return self.x_train[line_number]
  
  #Gets specified line in x_train where data are words
  def get_decoded_x_train_line(self, line_number):
    return " ".join(self.inverted_words_index[i] for i in self.x_train[line_number])

  #Gets y value of x_train's line number
  def get_value_for_y_train(self, line_number):
    return self.y_train[line_number]
  
  #Gets specified line in x_test where data are numbers
  def get_raw_x_test_line(self, line_number):
    return self.x_test[line_number]
  
  #Gets specified line in x_test where data are words
  def get_decoded_x_test_line(self, line_number):
    return " ".join(self.inverted_words_index[i] for i in self.x_test[line_number])

  #Gets y value of y_test line number
  def get_value_for_y_test(self, line_number):
    return self.y_test[line_number]
