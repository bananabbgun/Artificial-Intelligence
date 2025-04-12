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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        score = successorGameState.getScore()

        foodDis = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if foodDis:
            score += 1 / min(foodDis)

        ghostPos = [ghostState.getPosition() for ghostState in newGhostStates if ghostState.scaredTimer == 0]
        ghostDis = [manhattanDistance(newPos, ghostPos) for ghostPos in ghostPos]
        for dis in ghostDis:
            if dis < 4:  
                score -= 100

        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        def minValue(gameState, depth, agentIndex):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            value = float("inf")

            for action in gameState.getLegalActions(agentIndex):
                if agentIndex == gameState.getNumAgents() - 1:
                    value = min(value, maxValue(gameState.generateSuccessor(agentIndex, action), depth - 1))
                else:
                    value = min(value, minValue(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1))

            return value

        def maxValue(gameState, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            value = float("-inf")

            for action in gameState.getLegalActions(0):
                value = max(value, minValue(gameState.generateSuccessor(0, action), depth, 1))

            return value

        bestAction = None
        bestValue = float("-inf")
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = minValue(successor, self.depth, 1)
            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        def alpha_beta_search(state, depth, agent_index, alpha, beta):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            num_agents = state.getNumAgents()
            next_agent_index = (agent_index + 1) % num_agents
            
            # Check if we have completed one ply for all agents, and if so, decrement depth
            next_depth = depth - 1 if next_agent_index == 0 else depth

            # Delegate to the appropriate value function
            if agent_index == 0:  # Pacman, maximizing player
                return max_value(state, next_depth, agent_index, alpha, beta)
            else:  # Ghosts, minimizing players
                return min_value(state, next_depth, agent_index, alpha, beta)
        
        def max_value(state, depth, agent_index, alpha, beta):
            v = float("-inf")
            for action in state.getLegalActions(agent_index):
                successor = state.generateSuccessor(agent_index, action)
                v = max(v, alpha_beta_search(successor, depth, 1, alpha, beta))
                if v > beta:  
                    return v
                alpha = max(alpha, v)
            return v
        
        def min_value(state, depth, agent_index, alpha, beta):
            v = float("inf")
            for action in state.getLegalActions(agent_index):
                successor = state.generateSuccessor(agent_index, action)
                # Calculate the next agent's index, could be another ghost or Pacman
                next_agent_index = (agent_index + 1) % state.getNumAgents()
                v = min(v, alpha_beta_search(successor, depth, next_agent_index, alpha, beta))
                if v < alpha:  
                    return v
                beta = min(beta, v)
            return v

        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_action = None
        for action in gameState.getLegalActions(0):  # Actions for Pacman
            successor = gameState.generateSuccessor(0, action)
            score = alpha_beta_search(successor, self.depth, 1, alpha, beta)  # Start with the first ghost
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, best_score)  # Update alpha for the root node
        
        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimax(state, depth, agentIndex):
            # Check for terminal state or if depth limit has been reached
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            # Pacman's turn (Maximizer)
            if agentIndex == 0:
                return max_value(state, depth)
            # Ghosts' turn (Chance node)
            else:
                return exp_value(state, depth, agentIndex)

        def max_value(state, depth):
            # Initialize value
            v = float("-inf")
            # Iterate through possible actions and calculate the maximum value
            for action in state.getLegalActions(0):
                v = max(v, expectimax(state.generateSuccessor(0, action), depth, 1))
            return v

        def exp_value(state, depth, agentIndex):
            # Initialize value and get legal actions for the ghost
            v = 0
            legalActions = state.getLegalActions(agentIndex)
            # If there are no legal actions, return the evaluation function's result
            if not legalActions:
                return self.evaluationFunction(state)
            # Calculate the expected value
            for action in legalActions:
                # Probability of each action (uniform distribution)
                prob = 1.0 / len(legalActions)
                # Update the expected value
                v += prob * expectimax(state.generateSuccessor(agentIndex, action), 
                                       depth - 1 if agentIndex == state.getNumAgents() - 1 else depth, 
                                       (agentIndex + 1) % state.getNumAgents())
            return v

        # Begin expectimax
        bestAction = Directions.STOP
        bestValue = float("-inf")
        # Iterate through possible actions for Pacman to find the best one
        for action in gameState.getLegalActions(0):
            value = expectimax(gameState.generateSuccessor(0, action), self.depth, 1)
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    """
    if currentGameState.isWin():
        return float("inf")  # Maximize score if winning
    if currentGameState.isLose():
        return -float("inf")  # Minimize score if losing

    # Get the current score
    score = currentGameState.getScore()

    # Get new pacman position
    newPos = currentGameState.getPacmanPosition()
    
    # Evaluate distance to food
    newFood = currentGameState.getFood()
    foodList = newFood.asList()
    if foodList:
        minFoodDistance = min([manhattanDistance(newPos, food) for food in foodList])
        score += 1.0 / minFoodDistance

    # Evaluate distance to active and scared ghosts
    newGhostStates = currentGameState.getGhostStates()
    activeGhosts = [ghost for ghost in newGhostStates if not ghost.scaredTimer]
    scaredGhosts = [ghost for ghost in newGhostStates if ghost.scaredTimer]

    # We want to be closer to scared ghosts, and avoid active ghosts
    for ghost in activeGhosts:
        distance = manhattanDistance(newPos, ghost.getPosition())
        # More distance from active ghosts contributes positively
        score += max(8 - distance, 0) ** 2

    for ghost in scaredGhosts:
        distance = manhattanDistance(newPos, ghost.getPosition())
        # Closer distance to scared ghosts contributes positively
        score -= max(7 - distance, 0) ** 2

    # Evaluate remaining power pellets
    powerPelletsRemaining = len(currentGameState.getCapsules())
    score -= 4 * powerPelletsRemaining

    # Return the final score
    return score

# Abbreviation
better = betterEvaluationFunction



