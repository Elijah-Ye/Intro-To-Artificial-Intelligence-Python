'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

def k_nearest_neighbors(image, train_images, train_labels, k):
    '''
    Parameters:
    image - one image
    train_images - a list of N images
    train_labels - a list of N labels corresponding to the N images
    k - the number of neighbors to return

    Output:
    neighbors - 1-D array of k images, the k nearest neighbors of image
    labels - 1-D array of k labels corresponding to the k images
    '''
    #print(train_images[0].shape)
    dis_idx = []
    neighbors = np.zeros((k, train_images[0].shape[0]))
    labels = np.full(k, True)
    
    # I need to traverse the tran_images array
    for i in range(len(train_images)):
         # calculate the Euclidean distance between each training image and image
         dist = np.linalg.norm(train_images[i] - image)
         dis_idx.append((i, dist))
    
    dis_idx = sorted(dis_idx, key=lambda x: x[1])[:k]
    
    for x in range(k):
        neighbors[x] = train_images[dis_idx[x][0]]
        labels[x]=train_labels[dis_idx[x][0]]
    
    return neighbors, labels
    #raise RuntimeError('You need to write this part!')


def classify_devset(dev_images, train_images, train_labels, k):
    '''
    Parameters:
    dev_images (list) -M images
    train_images (list) -N images
    train_labels (list) -N labels corresponding to the N images
    k (int) - the number of neighbors to use for each dev image

    Output:
    hypotheses (list) -one majority-vote labels for each of the M dev images
    scores (list) -number of nearest neighbors that voted for the majority class of each dev image
    '''
    #print(dev_images.shape)
    hypotheses = []
    scores = []
    
    for i in range(len(dev_images)):
        neighors, labels = k_nearest_neighbors(dev_images[i], train_images, train_labels, k)
        x0 = np.sum(labels)
        x1 = len(labels) - x0
        if(x0 > x1):
            hypotheses.append(1)
            scores.append(x0)
        else:
            hypotheses.append(0)
            scores.append(x1)
        
    return hypotheses, scores
        
    
    #raise RuntimeError('You need to write this part!')


def confusion_matrix(hypotheses, references):
    '''
    Parameters:
    hypotheses (list) - a list of M labels output by the classifier
    references (list) - a list of the M correct labels

    Output:
    confusions (list of lists, or 2d array) - confusions[m][n] is 
    the number of times reference class m was classified as
    hypothesis class n.
    accuracy (float) - the computed accuracy
    f1(float) - the computed f1 score from the matrix
    '''
    tn = 0
    fp = 0
    fn = 0
    tp = 0
    
    for i in range(len(hypotheses)):
        if(hypotheses[i] == 0 and references[i] == False):
            tn += 1
        elif(hypotheses[i] == 1 and references[i] == False):
            fp += 1
        elif(hypotheses[i] == 0 and references[i] == True):
            fn += 1
        else: tp += 1
    
    confusions = np.array([[tn, fp], [fn, tp]])
    precision = 0.0
    recall = 0.0
    accuracy = 0.0
    f1 = 0.0
    
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    f1 = (2.0)/((1/recall) + (1/precision))
    
    return confusions, accuracy, f1
    
    

    #raise RuntimeError('You need to write this part!')
