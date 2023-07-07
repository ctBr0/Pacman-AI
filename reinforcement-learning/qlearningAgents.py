# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.Q_vals = util.Counter() # initialilze Counter "dictionary" of q_values

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        # the default value for the Counter "dictionary" is 0
        return self.Q_vals[(state, action)] # return the Q_value according to the current state and chosen action


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        curr_max = -2048.0
        legal_actions = self.getLegalActions(state)
        if (len(legal_actions) == 0): # no legal actions
            return 0.0
        for legal_action in legal_actions:
            Q_val = self.getQValue(state, legal_action)
            if Q_val > curr_max:
                curr_max = Q_val
        V_val = curr_max
        return V_val


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        action = None
        curr_max = -2048.0
        legal_actions = self.getLegalActions(state)
        for legal_action in legal_actions:
            Q_val = self.getQValue(state, legal_action)
            if Q_val > curr_max:
                action = legal_action
                curr_max = Q_val
        return action # the action with the greatest Q_value


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        is_random = util.flipCoin(self.epsilon) # return a bool
        if (is_random):
            action = random.choice(legalActions)
        else:
            action = self.getPolicy(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        # Q_pi(s,a) = (1-alpha)*Q_pi(s,a) + alpha*[R(s,a,s') + discount*Q_pi(s',pi(s'))]
        curr_Q = self.getQValue(state, action)
        new_Q = (1.0 - self.alpha) * curr_Q + self.alpha * (reward + self.discount * self.getQValue(nextState, self.computeActionFromQValues(nextState)))
        self.Q_vals[(state, action)] = new_Q


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        features = self.featExtractor.getFeatures(state, action) # (string, float) tuple list
        Q_val = 0.0
        for feature in features:
            Q_val += self.weights[feature] * features[feature]
        return Q_val


    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        # difference = [reward + discount * max(a') Q(s',a')] - Q(s,a)
        # weight_i = weight_i + alpha * difference * feature_i(s,a)
        features = self.featExtractor.getFeatures(state, action) # (string, float) tuple list
        diff = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)
        for feature in features:
            self.weights[feature] += self.alpha * diff * features[feature]


    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            print("alpha: " + str(self.alpha))
            print("discount rate: " + str(self.discount))
            print("epsilon: " + str(self.epsilon))
            print("numTraining: " + str(self.numTraining))
            features = self.featExtractor.getFeatures(state, self.getAction(state))
            for feature in features:
                print("feature weight: " + str(feature) + " " + str(self.weights[feature]))

            
