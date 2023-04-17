'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

epsilon = 1e-3

def compute_transition_matrix(model):
    '''
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    '''
    P = np.zeros([model.M, model.N, 4, model.M, model.N])
    for m in range(model.M):
        for n in range(model.N):
        
            if(model.T[m, n] == True):
                P[m, n, :, :, :] = 0
            else:
            # left
                if n-1 >= 0 and model.W[m, n-1] == False:
                    P[m, n, 0, m, n-1] = model.D[m, n, 0]
                elif (n-1 >= 0 and model.W[m, n-1] == True) or n - 1 < 0:
                    P[m, n, 0, m, n] += model.D[m, n, 0]
                # down
                if m + 1 < model.M and model.W[m + 1, n] == False:
                    P[m, n, 0, m + 1, n] = model.D[m, n, 1]
                elif (m + 1 < model.M and model.W[m + 1, n] == True) or m + 1 >= model.M:
                    P[m, n, 0, m, n] += model.D[m, n, 1]
                # up
                if m - 1 >= 0 and model.W[m -1, n] == False:
                    P[m, n, 0, m - 1, n] = model.D[m, n, 2]
                elif (m - 1 >= 0 and model.W[m -1, n] == True) or m - 1 < 0:
                    P[m, n, 0, m, n] += model.D[m, n, 2]
            
            # up
                if m - 1 >= 0 and model.W[m -1, n] == False:
                    P[m, n, 1, m - 1, n] = model.D[m, n, 0]
                elif (m - 1 >= 0 and model.W[m -1, n] == True) or m - 1 < 0:
                    P[m, n, 1, m, n] = model.D[m, n, 0]
                # left
                if n-1 >= 0 and model.W[m, n-1] == False:
                    P[m, n, 1, m, n-1] = model.D[m, n, 1]
                elif (n-1 >= 0 and model.W[m, n-1] == True) or n - 1 < 0:
                    P[m, n, 1, m, n] += model.D[m, n, 1]
                # right
                if n+1 < model.N and model.W[m, n+1] == False:
                    P[m, n, 1, m, n+1] = model.D[m, n, 2]
                elif (n+1 < model.N and model.W[m, n+1] == True) or n + 1 >= model.N:
                    P[m, n, 1, m, n] += model.D[m, n, 2]
                    
            # right
                if n+1 < model.N and model.W[m, n+1] == False:
                    P[m, n, 2, m, n+1] = model.D[m, n, 0]
                elif (n+1 < model.N and model.W[m, n+1] == True) or n + 1 >= model.N:
                    P[m, n, 2, m, n] += model.D[m, n, 0]
                # up
                if m - 1 >= 0 and model.W[m - 1, n] == False:
                    P[m, n, 2, m - 1, n] = model.D[m, n, 1]
                elif (m - 1 >= 0 and model.W[m -1, n] == True) or m - 1 < 0:
                    P[m, n, 2, m, n] += model.D[m, n, 1]
                # down
                if m + 1 < model.M and model.W[m + 1, n] == False:
                    P[m, n, 2, m + 1, n] = model.D[m, n, 2]
                elif (m + 1 < model.M and model.W[m + 1, n] == True) or m + 1 >= model.M:
                    P[m, n, 2, m, n] += model.D[m, n, 2]
            
            #down
                if m + 1 < model.M and model.W[m + 1, n] == False:
                    P[m, n, 3, m + 1, n] = model.D[m, n, 0]
                elif (m + 1 < model.M and model.W[m + 1, n] == True) or m + 1 >= model.M:
                    P[m, n, 3, m, n] += model.D[m, n, 0]
                # right
                if n+1 < model.N and model.W[m, n+1] == False:
                    P[m, n, 3, m, n+1] = model.D[m, n, 1]
                elif (n+1 < model.N and model.W[m, n+1] == True) or n + 1 >= model.N:
                    P[m, n, 3, m, n] += model.D[m, n, 1]
                # left
                if n-1 >= 0 and model.W[m, n-1] == False:
                    P[m, n, 3, m, n-1] = model.D[m, n, 2]
                elif (n-1 >= 0 and model.W[m, n-1] == True) or n - 1 < 0:
                    P[m, n, 3, m, n] += model.D[m, n, 2] 
    return P
                    
    #raise RuntimeError("You need to write this part!")

def update_utility(model, P, U_current):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()
    P - The precomputed transition matrix returned by compute_transition_matrix()
    U_current - The current utility function, which is an M x N array

    Output:
    U_next - The updated utility function, which is an M x N array
    '''
    U_next = np.zeros([model.M, model.N])
    for m in range(model.M):
        for n in range(model.N):
            if model.T[m,n] == True:
                U_next[m,n] = model.R[m, n]
            else:
                U_next[m][n] = model.R[m, n] + model.gamma * max(
                    (P[m, n, 0, :, :]*U_current).sum(), 
                    (P[m, n, 1, :, :]*U_current).sum(),
                    (P[m, n, 2, :, :]*U_current).sum(),
                    (P[m, n, 3, :, :]*U_current).sum())
    
    return U_next
    #raise RuntimeError("You need to write this part!")

def value_iteration(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    '''
    U = np.zeros([model.M, model.N])
    P = compute_transition_matrix(model)
    for i in range(100):
        U = update_utility(model, P, U)
        
    return U
    # raise RuntimeError("You need to write this part!")

if __name__ == "__main__":
    import utils
    model = utils.load_MDP('models/small.json')
    model.visualize()
    U = value_iteration(model)
    model.visualize(U)
