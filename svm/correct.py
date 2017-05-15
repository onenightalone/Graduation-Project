import codecs

def feature():
    y = 0
    ttl = 0
    with codecs.open('result_happy_feature.txt','r+',encoding='utf-8') as fin:
        for i in range(5000):
            r = fin.readline()
            if r == "[0]\n":
                y += 1
                ttl += 1
            else:
                ttl +=1

        print(y)
        print(ttl)

        correct = (y/ttl)*100
        print('percentage of correcting is' , correct, '%')

if __name__ == '__main__':
    feature()
