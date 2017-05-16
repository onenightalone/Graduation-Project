from gensim.models.keyedvectors import KeyedVectors
import codecs
import re
from numpy import *
import numpy
from dictionary_v3 import *
import copy

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

#load data
model = KeyedVectors.load_word2vec_format("wiki.cn.jian.vector",binary=True)

dictionary = get_dict()

#train data
x1 = []
y1 = []
dic = []

f = ['happy','angry','disgust','sad']

for n in range(len(f)):
    print 'Loading train data', str(n+1)
    file_name = 'train_'+f[n]+'.txt'
    with codecs.open(file_name,'r+',encoding='utf-8') as fin:
        label = n        
        for i in range(30000):
            if i%300 == 0:
                print str(i/300) + '%'
            content = fin.readline()
            if (i+1)%2 == 0:
                continue
            
            sentence = re.sub(r'[^\u4E00-\u9FA5 \n]', "", content)
            words = sentence.split(' ')

            sent_vec = array([], dtype = 'float32')
            for word in words:
                fq = 0
                if not word:
                    continue
                vector = array([], dtype = 'float32')
                if word in dictionary:
                    try:
                        vector = copy.deepcopy(model[word])#.astype(float64)
                        for i in range(len(vector)):
                            vector[i] *= dictionary[word]
                        fq += 1
                    except KeyError:
                        continue
                else:
                    continue
                is_inf = False
                for j in range(len(vector)):
                    if numpy.isinf(vector[j]) or vector[j]=='inf' or vector[j]=='-inf':
                    	is_inf = True
                    	break
                    else:
                        if fq == 1:
                            sent_vec = copy.deepcopy(vector)
                        else:
                            sent_vec = numpy.row_stack((sent_vec,vector))
                if is_inf:
                    continue
            if len(sent_vec) == 0:
                continue
            else:
                x1.append(sent_vec)
                y1.append(label)

x_train = numpy.asarray(x1)#.astype(float64)
y_train = numpy.asarray(y1)#.astype(float64)

#test data
x2 = []
y2 = []

for n in range(len(f)):
    print 'Loading test data', str(n+1）
    file_name = 'devtest_'+f[n]+'.txt'
    with codecs.open(file_name,'r+',encoding='utf-8') as fin:
        label = n        
        for i in range(10000):
            if i%100 == 0:
                print str(i/100) + '%'
            content = fin.readline()
            if (i+1)%2 == 0:
                continue
            
            sentence = re.sub(r'[^\u4E00-\u9FA5 \n]', "", content)
            words = sentence.split(' ')

            sent_vec = array([], dtype = 'float32')
            for word in words:
                fq = 0
                if not word:
                    continue
                vector = array([], dtype = 'float32')
                if word in dictionary:
                    try:
                        vector = copy.deepcopy(model[word])#.astype(float64)
                        #for i in range(len(vector)):
                            #vector[i] *= dictionary[word]
                        fq += 1
                    except KeyError:
                        continue
                else:
                    continue
                is_inf = False
                for j in range(len(vector)):
                    if numpy.isinf(vector[j]) or vector[j]=='inf' or vector[j]=='-inf':
                    	is_inf = True
                    	break
                    else:
                        if fq == 1:
                            sent_vec = copy.deepcopy(vector)
                        else:
                            sent_vec = numpy.row_stack((sent_vec,vector))
                if is_inf:
                    continue
            if len(sent_vec) == 0:
                continue
            else:
                x2.append(sent_vec)
                y2.append(label)

x_test = numpy.asarray(x2)#.astype(float64)
y_test = numpy.asarray(y2)#.astype(float64)

#train & predict
max_features = 10000000
maxlen = 150  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print 'Pad sequences (samples x time)'
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print 'x_train shape:', x_train.shape)
print 'x_test shape:', x_test.shape)

print 'Build model...'
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print 'Train...'
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print 'Test score:', score
print 'Test accuracy:', acc
