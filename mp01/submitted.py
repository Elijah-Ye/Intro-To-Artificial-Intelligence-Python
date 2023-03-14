'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np

def joint_distribution_of_word_counts(texts, word0, word1):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word to count
    word1 (str) - the second word to count

    Output:
    Pjoint (numpy array) - Pjoint[m,n] = P(X1=m,X2=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    '''
    max_word0_count = 0
    max_word1_count = 0
    for text in texts:
      if(text.count(word0) > max_word0_count):
        max_word0_count = text.count(word0)
      if(text.count(word1) > max_word1_count):
        max_word1_count = text.count(word1)
    
    array_size = max(max_word0_count, max_word1_count) + 1
    Pjoint = np.zeros((array_size,array_size))
    
    for text in texts:
      Pjoint[text.count(word0)][text.count(word1)] += 1
      
    Pjoint = Pjoint/Pjoint.sum()
      
    
    # raise RuntimeError('You need to write this part!')
    return Pjoint

def marginal_distribution_of_word_counts(Pjoint, index):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    index (0 or 1) - which variable to retain (marginalize the other) 

    Output:
    Pmarginal (numpy array) - Pmarginal[x] = P(X=x), where
      if index==0, then X is X0
      if index==1, then X is X1
    '''
    Pmarginal = np.zeros(Pjoint.shape[0])
    
    if index == 0:
      Pmarginal = np.sum(Pjoint, axis=1)
      
    if index == 1:
      Pmarginal = np.sum(Pjoint, axis=0)

    # raise RuntimeError('You need to write this part!')
    return Pmarginal
    
def conditional_distribution_of_word_counts(Pjoint, Pmarginal):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
    Pmarginal (numpy array) - Pmarginal[m] = P(X0=m)

    Outputs: 
    Pcond (numpy array) - Pcond[m,n] = P(X1=n|X0=m)
    '''
    Pcond = np.divide(np.transpose(Pjoint), Pmarginal)
    
    Pcond = np.transpose(Pcond)
    
    # raise RuntimeError('You need to write this part!')
    return Pcond

def mean_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)
    
    Outputs:
    mu (float) - the mean of X
    '''
    mu = 0
    # print(np.sum(P))
    for i in range(P.shape[0]):
      mu += P[i] * i
    
    # raise RuntimeError('You need to write this part!')
    return mu

def variance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)
    
    Outputs:
    var (float) - the variance of X
    '''
    
    mu = 0
    for i in range(P.shape[0]):
      mu += P[i] * i
      
    var = 0
    for i in range(P.shape[0]):
      var += (i - mu)*(i - mu)*(P[i])
    
    
    # raise RuntimeError('You need to write this part!')
    return var

def covariance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[m,n] = P(X0=m,X1=n)
    
    Outputs:
    covar (float) - the covariance of X0 and X1
    '''
    covar = 0.0
    x0 = np.sum(P, axis = 1)
    x1 = np.sum(P, axis = 0)
    
    mean0 = mean_from_distribution(x0)
    mean1 = mean_from_distribution(x1)
    
    for i in range(P.shape[0]):
      for j in range(P.shape[1]):
        covar += (i - mean0)*(j - mean1)*(P[i][j])
    
    # raise RuntimeError('You need to write this part!')
    return covar

def expectation_of_a_function(P, f):
    '''
    Parameters:
    P (numpy array) - joint distribution, P[m,n] = P(X0=m,X1=n)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       must be a real number for all values of (x0,x1)
       such that P(X0=x0,X1=x1) is nonzero.

    Output:
    expected (float) - the expected value, E[f(X0,X1)]
    '''
    expected = 0
    for i in range(P.shape[0]):
      for j in range(P.shape[1]):
        expected += f(i, j)*(P[i][j])
    
    
    # raise RuntimeError('You need to write this part!')
    return expected
    
