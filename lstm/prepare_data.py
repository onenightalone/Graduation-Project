from gensim.models.keyedvectors import KeyedVectors
import codecs
import re
from numpy import *
import numpy
from dictionary_v3 import *
import copy

#zero_vec = array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype(float32)
#for i in range(20):
    #zero_vec.append(0)
model = KeyedVectors.load_word2vec_format("wiki.cn.jian.vector",binary=True)

dictionary = get_dict()

#train data
x = []
y = []
dic = []

f = ['happy','angry','disgust','sad']

for n in range(len(f)):
    print('Loading train data', str(n+1))
    file_name = 'train_'+f[n]+'.txt'
    with codecs.open(file_name,'r+',encoding='utf-8') as fin:
        label = n        
        for i in range(30000):
            if i%300 == 0:
                print(str(i/300) + '%')
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
                            sent_vec = vector
                        else:
                            sent_vec = numpy.row_stack((sent_vec,vector))
                if is_inf:
                    continue
            if len(sent_vec) == 0:
                continue
            else:
                x.append(sent_vec)
                y.append(label)

x_train = numpy.asarray(x)#.astype(float64)
y_train = numpy.asarray(y)#.astype(float64)

#test data
x = []
y = []

for n in range(len(f)):
    print('Loading test data', str(n+1))
    file_name = 'devtest_'+f[n]+'.txt'
    with codecs.open(file_name,'r+',encoding='utf-8') as fin:
        label = n        
        for i in range(10000):
            if i%100 == 0:
                print(str(i/100) + '%')
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
                            sent_vec = vector
                        else:
                            sent_vec = numpy.row_stack((sent_vec,vector))
                if is_inf:
                    continue
            if len(sent_vec) == 0:
                continue
            else:
                x.append(sent_vec)
                y.append(label)

x_test = numpy.asarray(x)#.astype(float64)
y_test = numpy.asarray(y)#.astype(float64)

#save data
print('Saving data....')
numpy.save('x_train.npy',x_train)
numpy.save('y_train.npy',y_train)
numpy.save('x_test.npy',x_test)
numpy.save('y_test.npy',y_test)
