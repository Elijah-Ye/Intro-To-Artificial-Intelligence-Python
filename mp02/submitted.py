'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
from collections import Counter
import copy
import math

stopwords = set(["a","about","above","after","again","against","all","am","an","and","any","are","aren","'t","as","at","be","because","been","before","being","below","between","both","but","by","can","cannot","could","couldn","did","didn","do","does","doesn","doing","don","down","during","each","few","for","from","further","had","hadn","has","hasn","have","haven","having","he","he","'d","he","'ll","he","'s","her","here","here","hers","herself","him","himself","his","how","how","i","'m","'ve","if","in","into","is","isn","it","its","itself","let","'s","me","more","most","mustn","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan","she","she","'d","she","ll","she","should","shouldn","so","some","such","than","that","that","the","their","theirs","them","themselves","then","there","there","these","they","they","they","they","'re","they","this","those","through","to","too","under","until","up","very","was","wasn","we","we","we","we","we","'ve","were","weren","what","what","when","when","where","where","which","while","who","who","whom","why","why","with","won","would","wouldn","you","your","yours","yourself","yourselves"])

def create_frequency_table(train):
    '''
    Parameters:
    train (dict of list of lists) 
        - train[y][i][k] = k'th token of i'th text of class y

    Output:
    frequency (dict of Counters) 
        - frequency[y][x] = number of tokens of word x in texts of class y
    '''
    frequency = {}
    
    for y in train.keys():
        frequency[y] = Counter()
        for x in range(len(train[y])):
            for k in train[y][x]:
                frequency[y][k] += 1
    
                
    return frequency
    #raise RuntimeError("You need to write this part!")

def remove_stopwords(frequency):
    '''
    Parameters:
    frequency (dict of Counters) 
        - frequency[y][x] = number of tokens of word x in texts of class y

    Output:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in texts of class y,
          but only if x is not a stopword.
    '''
    nonstop = copy.deepcopy(frequency)
    
    for y in frequency:
        for x in frequency[y]:
            if x in stopwords:
                del nonstop[y][x]
    
    return nonstop
    # raise RuntimeError("You need to write this part!")

def laplace_smoothing(nonstop, smoothness):
    '''
    Parameters:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of x in y, if x not a stopword
    smoothness (float)
        - smoothness = Laplace smoothing hyperparameter

    Output:
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of x given y
        - likelihood[y]['OOV'] = likelihood of an out-of-vocabulary word given y

    Be careful that your vocabulary only counts words that occurred at least once
    in the training data for class y.
    '''
    likelihood = {}
    denominator = {}
    
    for y in nonstop.keys():
        denominator[y] = (sum(nonstop[y].values()) + smoothness*(len(nonstop[y].keys()) + 1))

    for y in nonstop.keys():
        likelihood[y] = {}
        for x in nonstop[y]:
            likelihood[y][x] = (nonstop[y][x] + smoothness)/denominator[y]
    
    for y in nonstop.keys():
        likelihood[y]['OOV'] = (smoothness)/denominator[y]
            
    return likelihood
        
    #raise RuntimeError("You need to write this part!")

def naive_bayes(texts, likelihood, prior):
    '''
    Parameters:
    texts (list of lists) -
        - texts[i][k] = k'th token of i'th text
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of x given y
    prior (float)
        - prior = the prior probability of the class called "pos"

    Output:
    hypotheses (list)
        - hypotheses[i] = class label for the i'th text
    '''
    hypotheses = []
    
    for text in texts:
        pos = math.log(prior)
        neg = math.log(1 - prior)
        for x in range(len(text)):
            if text[x] in stopwords: continue
            elif text[x] not in likelihood['neg']: neg += math.log(likelihood['neg']['OOV'])
            else: neg += math.log(likelihood['neg'][text[x]])
        for x in range(len(text)):
            if text[x] in stopwords: continue
            elif text[x] not in likelihood['pos']: pos += math.log(likelihood['pos']['OOV'])
            else: pos += math.log(likelihood['pos'][text[x]])
        
        if(pos > neg): hypotheses.append('pos')
        elif (neg > pos): hypotheses.append('neg')
        else: hypotheses.append('undecided')
            
    return hypotheses
    
    #raise RuntimeError("You need to write this part!")

def optimize_hyperparameters(texts, labels, nonstop, priors, smoothnesses):
    '''
    Parameters:
    texts (list of lists) - dev set texts
        - texts[i][k] = k'th token of i'th text
    labels (list) - dev set labels
        - labels[i] = class label of i'th text
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in class y, x not stopword
    priors (list)
        - a list of different possible values of the prior
    smoothnesses (list)
        - a list of different possible values of the smoothness

    Output:
    accuracies (numpy array, shape = len(priors) x len(smoothnesses))
        - accuracies[m,n] = dev set accuracy achieved using the
          m'th candidate prior and the n'th candidate smoothness
    '''
    
    accuracies = np.zeros((len(priors), len(smoothnesses)))
    
    for x in range(len(priors)):
        for y in range(len(smoothnesses)):
            likelihood = laplace_smoothing(nonstop, smoothnesses[y])
            hypotheses = naive_bayes(texts, likelihood, priors[x])
            count_correct = 0
            for (z,zhat) in zip(labels, hypotheses):
                if z==zhat:
                    count_correct += 1
                    
            accuracies[x][y] = count_correct / len(labels)
    
    return accuracies
    
    #raise RuntimeError("You need to write this part!")
                          
