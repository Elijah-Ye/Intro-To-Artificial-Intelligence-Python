'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not 
use other non-standard modules (including nltk). Some modules that might be helpful are 
already imported for you.
'''

import math
from collections import defaultdict, Counter
from math import log
import numpy as np
import pprint 

# define your epsilon for laplace smoothing here

def baseline(train, test):
    '''
    Implementation for the baseline tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    prediction = []
    wordTag = {}
    countTag = Counter()
    
    for sentence in train:
        for word_tag in sentence:
            word, tag = word_tag
            if word not in wordTag:
                wordTag[word] = Counter()
            wordTag[word][tag] += 1
            countTag[tag] += 1
    
    mostTag = max(countTag.keys(), key=(lambda key: countTag[key]))
    
    for sentence in test:
        predTags = []
        for word in sentence:  
        # in test there is no tags for each word
            if word in wordTag:
                # for seen word, we will use the tag that appears the most for this word
                bestTag = max(wordTag[word].keys(), key=lambda key:wordTag[word][key])  
                predTags.append((word, bestTag))
            else:
                # if we never seen this tag before, we will use the tag that appears the most in the training set
                predTags.append((word, mostTag))                   
        prediction.append(predTags)
    
    return prediction
    
    #raise NotImplementedError("You need to write this part!")


def viterbi(train, test):
    '''
    Implementation for the viterbi tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    k = 0.0001      # this is for laplace smoothing
    
    
    predictions = []
    tagTag = Counter()     
    countTagWord = Counter() 
    countTag = Counter()
    words = set()   # sets of word that have appeared in the training set
    tags = set()    # sets of tag that have appeared in the training set
    startTag = Counter()
    
    # step 1: count occurrences of tags, tag pairs, and tag_word pairs
    for sentence in train:
        startTag[sentence[0][1]] += 1
        for word_tag in sentence:
            w, t = word_tag
            countTagWord[word_tag] += 1
            countTag[t] += 1
            words.add(w)
            tags.add(t)
        for i in range(1, len(sentence)):
            cur_tag = sentence[i][1]
            prev_tag = sentence[i-1][1]
            tagTag[(prev_tag, cur_tag)] += 1
    
    # step 2 & 3: compute the log of the probabilities
    startProb = dict(startTag)
    for (tag, count) in startProb.items():
        startProb[tag] = log((count + k)/(len(train) + k*len(tags)))
    startProb_unknown = log(k/(len(train) + k*len(tags)))
    
    tagTagProb = dict(tagTag)
    for tag_a in tags:
        denominator = 0
        for tag_b in tags:
            if(tag_a, tag_b) in tagTagProb:
                denominator += tagTagProb[(tag_a, tag_b)]
        for tag_b in tags:
            if(tag_a, tag_b) in tagTagProb:
                tagTagProb[(tag_a, tag_b)] = log((tagTagProb[(tag_a, tag_b)] + k)/(denominator + k * len(tags)))
            else:
                tagTagProb[(tag_a, tag_b)] = log(k/(len(train) + k*len(tags)))
    tagTagProb_unknown = log(k/(len(train) + k*len(tags)))
    
    tagWordProb = dict(countTagWord)
    for word in words:
        for tag in tags:
            if(word, tag) in tagWordProb:
                tagWordProb[(word, tag)] = log((tagWordProb[(word, tag)] + k)/(countTag[tag] + k * (len(words) + 1)))
            else:
                tagWordProb[(word, tag)] = log(k/(len(train) + k * (len(words) + 1)))
    tagWordProb_unknown = log(k/(len(train) + k * (len(words) + 1)))
    
    # step 4 & 5: Construct the trellis and return the best path through trellis
    list_tags = list(tags)
    predictions = []
    for sentence in test:
        sentence_path = []
        viterbi_m = np.zeros((len(list_tags), len(sentence)))
        backpointer = np.zeros((len(list_tags), len(sentence)),dtype=int)
        for i in range(len(list_tags)):
            viterbi_m[i][0] = tagTagProb[('START', list_tags[i])]
            backpointer[i][0] = 3
        
        for j in range(1, len(sentence)):
            for i in range(len(list_tags)):
                max_v = -math.inf
                max_b = -math.inf
                for x in range(len(list_tags)):
                    if (sentence[j], list_tags[i]) in tagWordProb.keys():
                        cur_prob = viterbi_m[x][j - 1] + tagTagProb[(list_tags[x], list_tags[i])] + tagWordProb[(sentence[j], list_tags[i])]
                    else:
                        cur_prob = viterbi_m[x][j - 1] + tagTagProb[(list_tags[x], list_tags[i])] + tagWordProb_unknown
                    
                    if cur_prob > max_v :
                        max_v = cur_prob
                        max_b = x
                viterbi_m[i][j] = max_v
                backpointer[i][j] = max_b
        
        bestPathProb = -math.inf
        bestPathPointer = 0
        for i in range(len(list_tags)):
            cur_prob = viterbi_m[i][len(sentence) - 1]
            if cur_prob > bestPathProb:
                bestPathProb = cur_prob
                bestPathPointer = i
        
        for i in range(len(sentence)-1, -1, -1):
            if(i == 0):
                sentence_path.insert(0, (sentence[i], 'START'))
            else:
                sentence_path.insert(0, (sentence[i], list_tags[bestPathPointer]))
            bestPathPointer = backpointer[bestPathPointer][i]
        
        # print(sentence_path)
        # break
                
            
            
        predictions.append(sentence_path)
            
    return predictions
    

    #raise NotImplementedError("You need to write this part!")



def viterbi_ec(train, test):
    '''
    Implementation for the improved viterbi tagger.
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    k = 0.0001      # this is for laplace smoothing
    
    predictions = []
    tagTag = Counter()     
    countTagWord = Counter() 
    countTag = Counter()
    words = set()   # sets of word that have appeared in the training set
    tags = set()    # sets of tag that have appeared in the training set
    startTag = Counter()
    
    
    
    # step 1: count occurrences of tags, tag pairs, and tag_word pairs
    for sentence in train:
        startTag[sentence[0][1]] += 1
        for word_tag in sentence:
            w, t = word_tag
            countTagWord[word_tag] += 1
            countTag[t] += 1
            words.add(w)
            tags.add(t)
        for i in range(1, len(sentence)):
            cur_tag = sentence[i][1]
            prev_tag = sentence[i-1][1]
            tagTag[(prev_tag, cur_tag)] += 1
    
    # step 2 & 3: compute the log of the probabilities
    startProb = dict(startTag)
    for (tag, count) in startProb.items():
        startProb[tag] = log((count + k)/(len(train) + k*len(tags)))
    startProb_unknown = log(k/(len(train) + k*len(tags)))
    
    tagTagProb = dict(tagTag)
    for tag_a in tags:
        denominator = 0
        for tag_b in tags:
            if(tag_a, tag_b) in tagTagProb:
                denominator += tagTagProb[(tag_a, tag_b)]
        for tag_b in tags:
            if(tag_a, tag_b) in tagTagProb:
                tagTagProb[(tag_a, tag_b)] = log((tagTagProb[(tag_a, tag_b)] + k)/(denominator + k * len(tags)))
            else:
                tagTagProb[(tag_a, tag_b)] = log(k/(len(train) + k*len(tags)))
    tagTagProb_unknown = log(k/(len(train) + k*len(tags)))
    
    tagWordProb = dict(countTagWord)
    for word in words:
        for tag in tags:
            if(word, tag) in tagWordProb:
                tagWordProb[(word, tag)] = log((tagWordProb[(word, tag)] + k)/(countTag[tag] + k * (len(words) + 1)))
            else:
                tagWordProb[(word, tag)] = log(k/(len(train) + k * (len(words) + 1)))
    tagWordProb_unknown = log(k/(len(train) + k * (len(words) + 1)))
    
    # step 4 & 5: Construct the trellis and return the best path through trellis
    list_tags = list(tags)
    predictions = []
    for sentence in test:
        sentence_path = []
        viterbi_m = np.zeros((len(list_tags), len(sentence)))
        backpointer = np.zeros((len(list_tags), len(sentence)),dtype=int)
        for i in range(len(list_tags)):
            viterbi_m[i][0] = tagTagProb[('START', list_tags[i])]
            backpointer[i][0] = 3
        
        for j in range(1, len(sentence)):
            for i in range(len(list_tags)):
                max_v = -math.inf
                max_b = -math.inf
                for x in range(len(list_tags)):
                    if (sentence[j], list_tags[i]) in tagWordProb.keys():
                        cur_prob = viterbi_m[x][j - 1] + tagTagProb[(list_tags[x], list_tags[i])] + tagWordProb[(sentence[j], list_tags[i])]
                    else:
                        cur_prob = viterbi_m[x][j - 1] + tagTagProb[(list_tags[x], list_tags[i])] + tagWordProb_unknown
                    
                    if cur_prob > max_v :
                        max_v = cur_prob
                        max_b = x
                viterbi_m[i][j] = max_v
                backpointer[i][j] = max_b
        
        bestPathProb = -math.inf
        bestPathPointer = 0
        for i in range(len(list_tags)):
            cur_prob = viterbi_m[i][len(sentence) - 1]
            if cur_prob > bestPathProb:
                bestPathProb = cur_prob
                bestPathPointer = i
        
        for i in range(len(sentence)-1, -1, -1):
            sentence_path.insert(0, (sentence[i], list_tags[bestPathPointer]))
            bestPathPointer = backpointer[bestPathPointer][i]
       
    
        predictions.append(sentence_path)
    
    return predictions
    #raise NotImplementedError("You need to write this part!")



