from util import manhattanDistance
from game import Directions
import random, util
from operator import itemgetter

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions():
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.

    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


    return successorGameState.getScore()


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
    Your minimax agent (problem 1)

    The auto grader will check the running time of your algorithm. Friendly reminder: passing the auto grader
    does not necessarily mean that your algorithm is correct.
  """


  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

      It is recommended you have separate functions: value(), max_value(), and min_value() as in the slides
      and call these functions here to make your code understandable.
    """

    # BEGIN_YOUR_CODE (around 35 lines of code expected)
    def max_value(gameState, treeDepth, agentIndex):
        valueAction = (float("-inf"), "Stop")
        numAgents = gameState.getNumAgents()
        legalMoves = gameState.getLegalActions(agentIndex)
        for action in legalMoves:
            nextState = gameState.generateSuccessor(agentIndex, action)
            valueAction = max([valueAction, (value(nextState, treeDepth-1, (agentIndex+1)%numAgents), action)], key = itemgetter(0))
        return valueAction

    def min_value(gameState, treeDepth, agentIndex):
        valueAction = (float("inf"), "Stop")
        numAgents = gameState.getNumAgents()
        legalMoves = gameState.getLegalActions(agentIndex)
        for action in legalMoves:
            nextState = gameState.generateSuccessor(agentIndex, action)
            valueAction = min([valueAction, (value(nextState, treeDepth-1, (agentIndex+1)%numAgents), action)], key = itemgetter(0))
        return valueAction

    def value(gameState, treeDepth, agentIndex):
        if gameState.isLose() or gameState.isWin() or treeDepth == 0:
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return max_value(gameState, treeDepth, agentIndex)[0]
        else: return min_value(gameState, treeDepth, agentIndex)[0]

    numAgents = gameState.getNumAgents()
    val, action = max_value(gameState, self.depth * numAgents, 0)
    #print val
    return action
    #raise Exception("Not implemented yet")
    # END_YOUR_CODE

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)

    The auto grader will check the running time of your algorithm. Friendly reminder: passing the auto grader
    does not necessarily mean your algorithm is correct.
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction

      The same methods used in MinimaxAgent should also be useful here

      It is recommended you have separate functions: value(), max_value(), and min_value() as in the slides
      and call these functions here to make the code clear
    """

    # BEGIN_YOUR_CODE (around 45 lines of code expected)
    def max_value(gameState, treeDepth, agentIndex, alpha, beta):
        valueAction = (float("-inf"), "Stop")
        numAgents = gameState.getNumAgents()
        legalMoves = gameState.getLegalActions(agentIndex)
        for action in legalMoves:
            nextState = gameState.generateSuccessor(agentIndex, action)
            valueAction = max([valueAction, (value(nextState, treeDepth-1, (agentIndex+1)%numAgents, alpha, beta), action)], key = itemgetter(0))
            alpha = max(alpha, valueAction[0])
            if alpha > beta: return valueAction
        return valueAction

    def min_value(gameState, treeDepth, agentIndex, alpha, beta):
        valueAction = (float("inf"), "Stop")
        numAgents = gameState.getNumAgents()
        legalMoves = gameState.getLegalActions(agentIndex)
        for action in legalMoves:
            nextState = gameState.generateSuccessor(agentIndex, action)
            valueAction = min([valueAction, (value(nextState, treeDepth-1, (agentIndex+1)%numAgents, alpha, beta), action)], key = itemgetter(0))
            beta = min(beta, valueAction[0])
            if beta < alpha: return valueAction
        return valueAction

    def value(gameState, treeDepth, agentIndex, alpha, beta):
        if gameState.isLose() or gameState.isWin() or treeDepth == 0:
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return max_value(gameState, treeDepth, agentIndex, alpha, beta)[0]
        else: return min_value(gameState, treeDepth, agentIndex, alpha, beta)[0]

    numAgents = gameState.getNumAgents()
    val, action = max_value(gameState, self.depth * numAgents, 0, float("-inf"), float("inf"))
    #print val
    return action
    #raise Exception("Not implemented yet")
    # END_YOUR_CODE

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)

    The auto grader will check the running time of your algorithm. Friendly reminder: passing the auto grader
    does not necessarily mean your algorithm is correct.
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.

      The same methods used in MinimaxAgent should also be useful here

      It is recommended you have separate functions: value(), max_value(), and expect_value() as in the slides
      and call these functions here to make the code clear
    """

    # BEGIN_YOUR_CODE (around 35 lines of code expected)
    def max_value(gameState, treeDepth, agentIndex):
        valueAction = (float("-inf"), "Stop")
        numAgents = gameState.getNumAgents()
        legalMoves = gameState.getLegalActions(agentIndex)
        for action in legalMoves:
            nextState = gameState.generateSuccessor(agentIndex, action)
            valueAction = max([valueAction, (value(nextState, treeDepth-1, (agentIndex+1)%numAgents), action)], key = itemgetter(0))
        return valueAction

    def expect_value(gameState, treeDepth, agentIndex):
        val = 0.0
        numAgents = gameState.getNumAgents()
        legalMoves = gameState.getLegalActions(agentIndex)
        for action in legalMoves:
            nextState = gameState.generateSuccessor(agentIndex, action)
            val += value(nextState, treeDepth-1, (agentIndex+1)%numAgents)
        return float(val/len(legalMoves))

    def value(gameState, treeDepth, agentIndex):
        if gameState.isLose() or gameState.isWin() or treeDepth == 0:
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return max_value(gameState, treeDepth, agentIndex)[0]
        else: return expect_value(gameState, treeDepth, agentIndex)

    numAgents = gameState.getNumAgents()
    val, action = max_value(gameState, self.depth * numAgents, 0)
    #print val
    return action
    #raise Exception("Not implemented yet")
    # END_YOUR_CODE

def betterEvaluationFunction(currentGameState):
    """
        Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
        evaluation function (problem 4).
        DESCRIPTION:
        We took linear combination of different aspects like distance from food
        pellets, ghosts and capsules. If the pacman is far from the ghosts then
        the state has high score. Also, if it is closer to food pellets or the
        capsules, the state should have high score. Eating capsules is more
        beneficial than pellets because the ghosts get scared and pacman can
        move freely. Also, eating ghosts in scared state hugely increases the
        final score. So, if scared ghosts are close then the state has high
        score, this makes the pacman move towards teh scared ghost. Inverse of
        distance is used to give score. Food pellets distance has a positive
        weight of 10. Capsule have +250, ghosts have -250, and scared ghosts have
        a weight of +500. This linear combination is added to the game score at
        that state to get the final Evaluation function value. Points are
        deducted for number for food pellets and capsules remaining to
        encourage the pacman to eat more. There is additional +50 points for
        finishing all food pellets
    """
    # BEGIN_YOUR_CODE (around 50 lines of code expected)
    currPos = currentGameState.getPacmanPosition()
    currFood = currentGameState.getFood()
    currGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in currGhostStates]
    #print newScaredTimes[0]
    capsulesPos = currentGameState.getCapsules()
    dist2Foods = []
    dist2Ghosts = []
    dist2Caps = []
    score = currentGameState.getScore()
    addScore = 0.0
    for food in currFood.asList():
        foodDist = manhattanDistance(currPos, food)
        dist2Foods.append(foodDist)
        addScore += float(1.0/foodDist)

    for caps in capsulesPos:
        capsDist = manhattanDistance(currPos, caps)
        dist2Caps.append(capsDist)
        addScore += float(25.0/capsDist)

    for ghost in currentGameState.getGhostPositions():
        dist2Ghosts.append(manhattanDistance(currPos, ghost))

    for times in newScaredTimes:
        if times > 2: addScore +=  float(50.0/dist2Ghosts[newScaredTimes.index(times)])
        else: addScore +=  float(-25.0/(1 + dist2Ghosts[newScaredTimes.index(times)]))
    score += 10*float(addScore)
    score += -len(dist2Foods) - len(dist2Caps)
    if len(dist2Foods) == 0:
        score += 50
    #print score
    return score
    #raise Exception("Not implemented yet")
    # END_YOUR_CODE

# Abbreviation
better = betterEvaluationFunction
