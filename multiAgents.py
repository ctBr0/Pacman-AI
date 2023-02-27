# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        foodPos = newFood.asList()
        activeGhostPos = []
        for ghost in newGhostStates:
            if (ghost.scaredTimer == 0):
                pos = ghost.getPosition()
                activeGhostPos.append(pos)
        
        if (len(foodPos) == 0): # end game
            return 100000
        else:
            # closest food distance
            distToClosestFood = 10000
            aveFoodDist = 0
            for food in foodPos:
                foodDist = manhattanDistance(newPos, food)
                if (foodDist < distToClosestFood):
                    distToClosestFood = foodDist
                aveFoodDist = aveFoodDist + foodDist
            aveFoodDist = aveFoodDist / len(foodPos)

            # closest and average ghost distance
            distToClosestGhost = 10000
            aveGhostDist = 0
            for actGhost in activeGhostPos:
                actGhostDist = manhattanDistance(newPos, actGhost)
                if (actGhostDist < distToClosestGhost):
                    distToClosestGhost = actGhostDist
                aveGhostDist = aveGhostDist + actGhostDist

            if (len(activeGhostPos) == 0):
                aveGhostDist = 100000
            else:
                aveGhostDist = aveGhostDist / len(activeGhostPos)

            score = 2 * distToClosestGhost + 1 * aveGhostDist - 0 * aveFoodDist - distToClosestFood - 100 * len(foodPos) # messed around with the coefficients
            return score
        

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        """
        print(gameState.getLegalActions(0))
        print(gameState.getLegalActions(1))

        print(gameState.generateSuccessor(0, gameState.getLegalActions(0)[0]))
        print(gameState.getNumAgents())
        print(gameState.isWin())
        """

        def minimax(gameState, currDepth, agentIndex, num_ghosts):
            if (currDepth == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)
            if (agentIndex == 0): # agent is Pacman
                maxVal = -100000.0
                legalActions = gameState.getLegalActions(agentIndex)
                for action in legalActions:
                    succ = gameState.generateSuccessor(agentIndex, action)
                    val = minimax(succ, currDepth + 1, agentIndex + 1, num_ghosts)
                    maxVal = max(maxVal, val)
                return maxVal
            else: # agent is ghost
                minVal = 100000.0
                legalActions = gameState.getLegalActions(agentIndex)
                for action in legalActions:
                    succ = gameState.generateSuccessor(agentIndex, action)
                    if (agentIndex == num_ghosts):
                        val = minimax(succ, currDepth + 1, 0, num_ghosts)
                    else:
                        val = minimax(succ, currDepth + 1, agentIndex + 1, num_ghosts)
                    minVal = min(minVal, val)
                return minVal
        
        maxVal = -100000.0
        currDepth = 0
        legalActions = gameState.getLegalActions(0)
        for action in legalActions:
            succ = gameState.generateSuccessor(0, action)
            val = minimax(succ, currDepth + 1, 1, gameState.getNumAgents() - 1)
            if val > maxVal:
                maxVal = val
                chosenAction = action

        return chosenAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        def alphabeta(gameState, currDepth, agentIndex, num_ghosts, alpha, beta):
            if (currDepth == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)
            if (agentIndex == 0): # agent is Pacman
                maxVal = -100000.0
                legalActions = gameState.getLegalActions(agentIndex)
                for action in legalActions:
                    succ = gameState.generateSuccessor(agentIndex, action)
                    val = alphabeta(succ, currDepth + 1, agentIndex + 1, num_ghosts, alpha, beta)
                    maxVal = max(maxVal, val)
                    alpha = max(alpha, val)
                    if beta < alpha:
                        break
                return maxVal
            else: # agent is ghost
                minVal = 100000.0
                legalActions = gameState.getLegalActions(agentIndex)
                for action in legalActions:
                    succ = gameState.generateSuccessor(agentIndex, action)
                    if (agentIndex == num_ghosts):
                        val = alphabeta(succ, currDepth + 1, 0, num_ghosts, alpha, beta)
                    else:
                        val = alphabeta(succ, currDepth + 1, agentIndex + 1, num_ghosts, alpha, beta)
                    minVal = min(minVal, val)
                    beta = min(beta, val)
                    if beta < alpha:
                        break
                return minVal
        
        maxVal = -100000.0
        currDepth = 0
        alpha = -100000.0
        beta = 100000.0
        legalActions = gameState.getLegalActions(0)
        for action in legalActions:
            succ = gameState.generateSuccessor(0, action)
            val = alphabeta(succ, currDepth + 1, 1, gameState.getNumAgents() - 1, alpha, beta)
            if val > maxVal:
                maxVal = val
                chosenAction = action
            if maxVal > beta:
                return maxVal
            alpha = max(alpha, maxVal)

        return chosenAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(gameState, currDepth, agentIndex, num_ghosts):
            if (currDepth == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)
            if (agentIndex == 0): # agent is Pacman
                maxVal = -100000.0
                legalActions = gameState.getLegalActions(agentIndex)
                for action in legalActions:
                    succ = gameState.generateSuccessor(agentIndex, action)
                    val = expectimax(succ, currDepth + 1, agentIndex + 1, num_ghosts)
                    maxVal = max(maxVal, val)
                return maxVal
            else: # agent is ghost
                minVal = 100000.0
                legalActions = gameState.getLegalActions(agentIndex)
                minVal = 0
                for action in legalActions:
                    succ = gameState.generateSuccessor(agentIndex, action)
                    if (agentIndex == num_ghosts):
                        minVal += expectimax(succ, currDepth + 1, 0, num_ghosts)
                    else:
                        minVal += expectimax(succ, currDepth + 1, agentIndex + 1, num_ghosts)
                return minVal / len(legalActions) # average of all the actions combined
        
        maxVal = -100000.0
        currDepth = 0
        legalActions = gameState.getLegalActions(0)
        for action in legalActions:
            succ = gameState.generateSuccessor(0, action)
            val = expectimax(succ, currDepth + 1, 1, gameState.getNumAgents() - 1)
            if val > maxVal:
                maxVal = val
                chosenAction = action

        return chosenAction
       

