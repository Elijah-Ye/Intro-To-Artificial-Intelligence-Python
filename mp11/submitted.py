'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import random
import numpy as np
import torch
import torch.nn as nn

class q_learner():
    def __init__(self, alpha, epsilon, gamma, nfirst, state_cardinality):
        '''
        Create a new q_learner object.
        Your q_learner object should store the provided values of alpha,
        epsilon, gamma, and nfirst.
        It should also create a Q table and an N table.
        Q[...state..., ...action...] = expected utility of state/action pair.
        N[...state..., ...action...] = # times state/action has been explored.
        Both are initialized to all zeros.
        Up to you: how will you encode the state and action in order to
        define these two lookup tables?  The state will be a list of 5 integers,
        such that 0 <= state[i] < state_cardinality[i] for 0 <= i < 5.
        The action will be either -1, 0, or 1.
        It is up to you to decide how to convert an input state and action
        into indices that you can use to access your stored Q and N tables.
        
        @params:
        alpha (scalar) - learning rate of the Q-learner
        epsilon (scalar) - probability of taking a random action
        gamma (scalar) - discount factor        
        nfirst (scalar) - exploring each state/action pair nfirst times before exploiting
        state_cardinality (list) - cardinality of each of the quantized state variables

        @return:
        None
        '''
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma 
        self.nfirst = nfirst
        
        self.Q = np.zeros((state_cardinality[0], state_cardinality[1], state_cardinality[2], state_cardinality[3], state_cardinality[4], 3))
        self.N = np.zeros((state_cardinality[0], state_cardinality[1], state_cardinality[2], state_cardinality[3], state_cardinality[4], 3))
        
        
        #raise RuntimeError('You need to write this!')

    def report_exploration_counts(self, state):
        '''
        Check to see how many times each action has been explored in this state.
        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        explored_count (array of 3 ints): 
          number of times that each action has been explored from this state.
          The mapping from actions to integers is up to you, but there must be three of them.
        '''
        explored_count = np.zeros(3)
        explored_count[0] = self.N[state[0]][state[1]][state[2]][state[3]][state[4]][0]
        explored_count[1] = self.N[state[0]][state[1]][state[2]][state[3]][state[4]][1]
        explored_count[2] = self.N[state[0]][state[1]][state[2]][state[3]][state[4]][2]
        
        return explored_count
        
        #raise RuntimeError('You need to write this!')

    def choose_unexplored_action(self, state):
        '''
        Choose an action that has been explored less than nfirst times.
        If many actions are underexplored, you should choose uniformly
        from among those actions; don't just choose the first one all
        the time.
        
        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
           These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        action (scalar): either -1, or 0, or 1, or None
          If all actions have been explored at least n_explore times, return None.
          Otherwise, choose one uniformly at random from those w/count less than n_explore.
          When you choose an action, you should increment its count in your counter table.
        '''
        action_list = []
        if(self.N[state[0]][state[1]][state[2]][state[3]][state[4]][0] < self.nfirst):
          action_list.append(0)
        if(self.N[state[0]][state[1]][state[2]][state[3]][state[4]][1] < self.nfirst):
          action_list.append(1)
        if(self.N[state[0]][state[1]][state[2]][state[3]][state[4]][2] < self.nfirst):
          action_list.append(-1)
        
        if(len(action_list) == 0): return None
        else:
          action = action_list[random.randrange(len(action_list))]
          self.N[state[0]][state[1]][state[2]][state[3]][state[4]][action] = self.N[state[0]][state[1]][state[2]][state[3]][state[4]][action] + 1
          return action
         
         
    
        #raise RuntimeError('You need to write this!')

    def report_q(self, state):
        '''
        Report the current Q values for the given state.
        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        Q (array of 3 floats): 
          reward plus expected future utility of each of the three actions. 
          The mapping from actions to integers is up to you, but there must be three of them.
        '''
        current_Q = np.zeros(3)
        current_Q[0] = self.Q[state[0]][state[1]][state[2]][state[3]][state[4]][0]
        current_Q[1] = self.Q[state[0]][state[1]][state[2]][state[3]][state[4]][1]
        current_Q[2] = self.Q[state[0]][state[1]][state[2]][state[3]][state[4]][2]
        
        return current_Q
        #raise RuntimeError('You need to write this!')

    def q_local(self, reward, newstate):
        '''
        The update to Q estimated from a single step of game play:
        reward plus gamma times the max of Q[newstate, ...].
        
        @param:
        reward (scalar float): the reward achieved from the current step of game play.
        newstate (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].
        
        @return:
        Q_local (scalar float): the local value of Q
        '''
        Q_local = reward + self.gamma * max(self.report_q(newstate))
        return Q_local
        #raise RuntimeError('You need to write this!')

    def learn(self, state, action, reward, newstate):
        '''
        Update the internal Q-table on the basis of an observed
        state, action, reward, newstate sequence.
        
        @params:
        state: a list of 5 numbers: ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle.
        action: an integer, one of -1, 0, or +1
        reward: a reward; positive for hitting the ball, negative for losing a game
        newstate: a list of 5 numbers, in the same format as state
        
        @return:
        None
        '''
        self.Q[state[0]][state[1]][state[2]][state[3]][state[4]][action] = self.Q[state[0]][state[1]][state[2]][state[3]][state[4]][action] + self.alpha * (self.q_local(reward, newstate) - self.Q[state[0]][state[1]][state[2]][state[3]][state[4]][action])
        
  
      
        #raise RuntimeError('You need to write this!')
    
    def save(self, filename):
        '''
        Save your Q and N tables to a file.
        This can save in any format you like, as long as your "load" 
        function uses the same file format.  We recommend numpy.savez,
        but you can use something else if you prefer.
        
        @params:
        filename (str) - filename to which it should be saved
        @return:
        None
        '''
        np.savez(filename, Q=self.Q, N=self.N)
        
        #raise RuntimeError('You need to write this!')
        
    def load(self, filename):
        '''
        Load the Q and N tables from a file.
        This should load from whatever file format your save function
        used.  We recommend numpy.load, but you can use something
        else if you prefer.
        
        @params:
        filename (str) - filename from which it should be loaded
        @return:
        None
        '''
        data = np.load(filename)
        self.Q = data['Q']
        self.N = data['N']
        
        #raise RuntimeError('You need to write this!')
        
    def exploit(self, state):
        '''
        Return the action that has the highest Q-value for the current state, and its Q-value.
        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        action (scalar int): either -1, or 0, or 1.
          The action that has the highest Q-value.  Ties can be broken any way you want.
        Q (scalar float): 
          The Q-value of the selected action
        '''
        q_current = list(self.report_q(state))
        action = q_current.index(max(q_current))
        if(action == 2):
          return -1, q_current[action]
        else:
          return action, q_current[action]
        #raise RuntimeError('You need to write this!')
    
    def act(self, state):
        '''
        Decide what action to take in the current state.
        If any action has been taken less than nfirst times, then choose one of those
        actions, uniformly at random.
        Otherwise, with probability epsilon, choose an action uniformly at random.
        Otherwise, choose the action with the best Q(state,action).
        
        @params: 
        state: a list of 5 integers: ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].
       
        @return:
        -1 if the paddle should move upward
        0 if the paddle should be stationary
        1 if the paddle should move downward
        '''
        action_list = []
        if(self.N[state[0]][state[1]][state[2]][state[3]][state[4]][0] < self.nfirst):
          action_list.append(0)
        if(self.N[state[0]][state[1]][state[2]][state[3]][state[4]][1] < self.nfirst):
          action_list.append(1)
        if(self.N[state[0]][state[1]][state[2]][state[3]][state[4]][2] < self.nfirst):
          action_list.append(-1)
        
        if(len(action_list) == 0): 
          test = np.random.choice(np.arange(0, 2), p=[self.epsilon, 1 - self.epsilon])
          if(test == 1):
            q_current = list(self.report_q(state))
            action = q_current.index(max(q_current))
            self.N[state[0]][state[1]][state[2]][state[3]][state[4]][action] = self.N[state[0]][state[1]][state[2]][state[3]][state[4]][action] + 1
            if(action == 2): return -1
            else: return action
          else:
            action_list = [0, 1, -1]
            action = action_list[random.randrange(len(action_list))]
            self.N[state[0]][state[1]][state[2]][state[3]][state[4]][action] = self.N[state[0]][state[1]][state[2]][state[3]][state[4]][action] + 1
            return action
            
        else:
          action = action_list[random.randrange(len(action_list))]
          self.N[state[0]][state[1]][state[2]][state[3]][state[4]][action] = self.N[state[0]][state[1]][state[2]][state[3]][state[4]][action] + 1
          return action
        
        #raise RuntimeError('You need to write this!')

class deep_q():
    def __init__(self, alpha, epsilon, gamma, nfirst):
        '''
        Create a new deep_q learner.
        Your q_learner object should store the provided values of alpha,
        epsilon, gamma, and nfirst.
        It should also create a deep learning model that will accept
        (state,action) as input, and estimate Q as the output.
        
        @params:
        alpha (scalar) - learning rate of the Q-learner
        epsilon (scalar) - probability of taking a random action
        gamma (scalar) - discount factor
        nfirst (scalar) - exploring each state/action pair nfirst times before exploiting

        @return:
        None
        '''
        raise RuntimeError('You need to write this!')

    def act(self, state):
        '''
        Decide what action to take in the current state.
        You are free to determine your own exploration/exploitation policy -- 
        you don't need to use the epsilon and nfirst provided to you.
        
        @params: 
        state: a list of 5 floats: ball_x, ball_y, ball_vx, ball_vy, paddle_y.
       
        @return:
        -1 if the paddle should move upward
        0 if the paddle should be stationary
        1 if the paddle should move downward
        '''
        raise RuntimeError('You need to write this!')
        
    def learn(self, state, action, reward, newstate):
        '''
        Perform one iteration of training on a deep-Q model.
        
        @params:
        state: a list of 5 floats: ball_x, ball_y, ball_vx, ball_vy, paddle_y
        action: an integer, one of -1, 0, or +1
        reward: a reward; positive for hitting the ball, negative for losing a game
        newstate: a list of 5 floats, in the same format as state
        
        @return:
        None
        '''
        raise RuntimeError('You need to write this!')
        
    def save(self, filename):
        '''
        Save your trained deep-Q model to a file.
        This can save in any format you like, as long as your "load" 
        function uses the same file format.
        
        @params:
        filename (str) - filename to which it should be saved
        @return:
        None
        '''
        raise RuntimeError('You need to write this!')
        
    def load(self, filename):
        '''
        Load your deep-Q model from a file.
        This should load from whatever file format your save function
        used.
        
        @params:
        filename (str) - filename from which it should be loaded
        @return:
        None
        '''
        raise RuntimeError('You need to write this!')
