import itertools
import numpy as np
from numpy import linalg
from scipy import stats
import pylab as pl
from sklearn import svm, linear_model, cross_validation
import sys
from pyspark import SparkContext
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

##read data
sc = SparkContext(appName="svm")
filename = sys.argv[1] ##filename
data = sc.textFile(filename)\
        .map(lambda x: x.split(','))
itemId = data.map(lambda x: x[0])\
        .collect()
vec = data.map(lambda x: map(lambda xx: float(xx), x[1:]))\
        .collect()

vec = np.array(vec)
vec10 = vec[:10]
vec90 = vec[10:]

######pair the training data######
#response = [1,-1]*10
response = [1]*10
neg_ids = map(lambda x: int(x)-1, sys.argv[2].split(','))
print neg_ids
for i in neg_ids:
    response[i] = -1
num_odd_even = dict(zip(range(10), response))
X_train = []
y_train = []
comb = itertools.combinations(range(10), 2)
for (a, b) in comb:
    diff = num_odd_even[a] - num_odd_even[b]
    X_train.append(vec[a] - vec [b])
    if diff == -2:
        y_train.append(-1)
    else:
        y_train.append(1)
X_train = np.array(X_train)
y_train = np.array(y_train)
##################################

######train data##calculate w#####
clf = svm.SVC(kernel='linear', C=.1)
clf.fit(X_train, y_train)
coef = clf.coef_.ravel() / linalg.norm(clf.coef_)

#rank data
scores = np.dot(vec[10:], np.array(coef).T).reshape(1,90)
rank = scores.argsort().argsort()

scores = np.insert(scores.reshape(90, 1),0,itemId[10:], axis = 1)
#order = scores[(-1*scores[:, 1]).argsort()]  
#items = ','.join(map(lambda x: str(int(x)), list(order[:,0])))
#out = sys.argv[3]
#open(out, 'w').write(items)

#get the data from als
item10 = itemId[:10]
item90 = itemId[10:]
item90_als = open(sys.argv[3], 'r').readlines()[0]\
        .strip().split(',')[1:]
for each in item10:
    item90_als.remove(each)

rank2 = []
for eachitem in item90_als:
    rank2.append(item90.index(str(eachitem)))

#evalulation
#calculate similarity#
cos = cosine_similarity(vec10, vec90)


for i in neg_ids:
    xaxis = range(1, 91)
    y1 = list(cos[i])   #原始排序
    y2 = np.array(y1)[(-1*scores[:,1]).argsort()] #svm排序
    y3 = np.array(y1)[rank2]  #als排序
    pl.plot(xaxis, y1, 'g--', xaxis, y2, 'b-', xaxis, y3, 'r-')
    pl.show()
    
#def mpr(x, y):
#    zipdata = zip(x, y)
#    numerator = np.sum(map(lambda x: x[0]*x[1]/89.0, zipdata))
#    denumrator = np.sum(x)
#    result = numerator / denumrator
#    return result
#
#mpr10 = []
#mpr10_learn = []
#for i in range(10):
#    rang = range(90)
#    rang.reverse()
#    mpr10.append(mpr(cos[i], rang))
#    mpr10_learn.append(mpr(cos[i], rank[0]))
#
#cols = ['gray']*10
#for i in neg_ids:
#    cols[i] = 'red'
#diff =  np.array(mpr10_learn) - np.array(mpr10)
#pl.bar(range(1, 11), diff, color = cols)
#pl.show()
