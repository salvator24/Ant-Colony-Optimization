import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mutual_info_score
import operator
import timeit

train = pd.read_csv('dataset/kddtrain_2class_normalized.csv')

train_y = train['attack_name']
train_x = train.drop(['attack_name'], axis = 1)
train_array = np.array(train_x)

#x: number of ants to be initialized
def init_subset(x, size):
    choice = np.zeros((x, 41))
    for i in range(x):
        #size = np.random.randint(10,39)
        choice_init = np.random.choice(np.arange(0, 41, 1), size=size, replace=False)
        choice[i][choice_init] = 1
    return choice

def fitness(particle):
    X = train_array[:, np.nonzero(particle)][:,0]
    clf = LogisticRegression(n_jobs=-4)
    return cross_val_score(clf, X, train_y, cv=3, scoring='accuracy').mean()

def mrmr(i, t):
    mi = []
    for a in np.nonzero(t)[0]:
        mi.append(mutual_info_score(train.ix[:,i], train.ix[:,a]))
    mi = sum(mi)
    mi = mutual_info_score(train.ix[:,i], train['attack_name'])-((1/len(np.nonzero(t)[0]))* mi)
    return mi

def usm(t, p):
    n=0.1
    k=0.2
    off = np.random.choice(np.nonzero(t)[0], size=p, replace=False)
    t[off] = 0
    for itr in range(p):
        usm_val={}
        for i in np.nonzero(t==0)[0]:
            m_r = mrmr(i, t)
            usm_val[i] =(np.power(phero[i],n)*np.power(m_r,k))
        s = sum(usm_val.values())
        for c in usm_val.items():
            usm_val[c[0]] = c[1]/s
        t[max(usm_val.items(), key=operator.itemgetter(1))[0]] = 1

num_ants = 100
num_itr = 15
phero = np.ones((1,41))

k=50
k_step=3
w=0.1
w_step=0.06

p=3
g=10

size = 22

fittest_accuracy = 0
old_accuracy = np.zeros(num_ants)

output = open('output/output.txt', 'w')

#intialize intial ant population
subset = init_subset(num_ants, size)

for i in range(num_itr):
    print(i)
    
    output.write(str(i))

    start = timeit.default_timer()
    
    #calculate accuracy for each ants
    accuracy = []
    for j in range(num_ants):
        accuracy.append(fitness(subset[j]))
    
    new_accuracy = np.array(accuracy)
    
    del_accuracy = new_accuracy - old_accuracy
    
    old_accuracy = np.array(accuracy)

    max_accuracy = max(accuracy)
    
    output.write(str(max_accuracy))
    output.write(str(fittest_accuracy))
	


    #find k best
    index_top = np.array(accuracy).argsort()[-k:][::-1]
    k_best = subset[index_top,:]
    
    #find g best
    g_best = subset[index_top,:]
    np.place(g_best, g_best==0, -1)
    
    #fittest ant
    if max_accuracy > fittest_accuracy:
        fittest_accuracy = max_accuracy
        fittest_ant = subset[index_top[0],:]

    '''

    #global importance update
    phero += w * np.dot(g_best.T, np.array(accuracy)[index_top])/k
    
    #calculate the change in pheromone 
    del_t = (np.amax(1-np.array(accuracy)) - (1-np.array(accuracy))[index_top]) \
    / np.amax((np.amax(1-np.array(accuracy)) - (1-np.array(accuracy))[index_top]))
    
    #update the pheromone according to kbest
    delta = np.multiply(k_best, del_t.reshape(k,1))
    phero += np.sum(delta, axis=0)
    
    '''

    #pheromone reenforcement
    del_t_reward = np.dot(g_best[np.where(del_accuracy[index_top]>=0)].T, np.array(accuracy)[np.where(del_accuracy[index_top]>=0)])/k
    del_t_penalty = np.dot(g_best[np.where(del_accuracy[index_top]<0)].T, np.array(accuracy)[np.where(del_accuracy[index_top]<0)])/k
    phero = w * (del_t_reward - del_t_penalty)


    #intialize next ant population
    new_subset = init_subset(num_ants-k, size)
    
    #evaluate USM
    for t in k_best:
        usm(t, p)
    
    subset = np.concatenate((k_best, new_subset), axis=0)
    
    k=k+k_step
    w=w+w_step
    
    end = timeit.default_timer()
    
    output.write(str(end - start))
    print("execution time: ", end - start)


high = np.nonzero(subset[10])[0]

output.write(str(high))

test = pd.read_csv('dataset/KDDTest+_normalized_2.csv')
test_y = test['attack_name']
test_x = test.drop(['attack_name'], axis = 1)

log_reg = LogisticRegression(n_jobs=-1)
log_reg.fit(train_x.ix[:,high], train_y)
pred = log_reg.predict(test_x.ix[:,high])

from sklearn.metrics import accuracy_score
output.write(str(accuracy_score(test_y, pred)))
output.close()
