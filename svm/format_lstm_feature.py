from gensim.models.keyedvectors import KeyedVectors
import gensim
import codecs
import re
from sklearn import svm
from sklearn.externals import joblib
from numpy import *
import numpy
from dictionary_v3 import *
import math
import sys
import copy

x = []
y = []

#zero_vec = array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype(float32)
#for i in range(20):
    #zero_vec.append(0)
model = KeyedVectors.load_word2vec_format("wiki.cn.jian.vector",binary=True)

#with codecs.open('知网评价情感词语.txt','r',encoding='utf-8') as fin:
    #dictionary = list(map(str.strip, fin.readlines()))

dictionary = get_dict()

f = ['happy','angry','disgust','sad']

for n in range(len(f)):
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
                        #vector = ','.join([str(i) for i in model[word]])
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
            #fout.write(label + str(values) + '\n')
            #s_v = numpy.asarray(sent_vec)#.astype(float64)
            if len(sent_vec) == 0:
                continue
            else:
                x.append(sent_vec)
                y.append(label)
            
'''
#统一维度
max_len = -1
for i in range(len(x)):
    if len(x[i]) > max_len:
        max_len = len(x[i])

print(max_len)

tmp = []
for i in range(len(x)):
    if len(x[i]) < max_len:
        while len(x[i]) < max_len:
            x[i] = numpy.row_stack((x[i],zero_vec))
    tmp.append(x[i])
'''
#print(x)

X = numpy.asarray(x)#.astype(float64)
Y = numpy.asarray(y)#.astype(float64)


clf = svm.SVC()  
clf.fit(X, Y)
joblib.dump(clf, 'sentiment_feature_strength.pkl')
