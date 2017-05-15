from gensim.models.keyedvectors import KeyedVectors
import gensim
import codecs
import re
from sklearn import svm
from sklearn.externals import joblib
from numpy import *
import numpy
import copy

model = KeyedVectors.load_word2vec_format("wiki.cn.jian.vector",binary=True)
clf = joblib.load("sentiment_feature_strength.pkl")

with codecs.open('知网评价情感词语.txt','r',encoding='utf-8') as fin:
    dictionary = list(map(str.strip, fin.readlines()))

zero_vec = array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype(float32)

f = ['happy','angry','disgust','sad']

for n in range(len(f)):
    print('Predicting ', f[n])
    fout_name = 'result_'+f[n]+'_feature.txt
    fin_name = 'devtest_'+f[n]+'.txt
    with codecs.open(fout_name,'w',encoding='utf-8') as fout:
        with codecs.open(fin_name,'r+',encoding='utf-8') as fin:
            for i in range(10000):
                if i%100 == 0:
                    print(str(i/100) + '%')
                content = fin.readline()
                if (i+1)%2 == 0:
                        continue
                sentence = re.sub(r'[^\u4E00-\u9FA5 \n]', "", content)
                words = sentence.split(' ')

                #sent_vec = array([])
                for word in words:
                    fq = 0
                    if not word:
                        continue
                    vector = array([])
                    if word in dictionary:
                        try:
                            #vector = ','.join([str(i) for i in model[word]])
                            vector = copy.deepcopy(model[word])#.astype(float64)
                            fq += 1
                        except KeyError:
                            #vector = ','.join([str(i) for i in zero_vec])
                            #vector = zero_vec
                            continue
                    else:
                        #vector = ','.join([str(i) for i in zero_vec])
                        #vector = zero_vec
                        continue
                    if fq == 1:
                        sent_vec = vector
                    else:
                        sent_vec = numpy.row_stack((sent_vec,vector))

                fout.write(str(clf.predict(sent_vec)))
                fout.write("\n")
