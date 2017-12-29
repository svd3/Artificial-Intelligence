import collections, util, math, random
from operator import itemgetter

############################################################
# Problem 3.1.1

def computeQ(mdp, V, state, action):
    """
    Return Q(state, action) based on V(state).  Use the properties of the
    provided MDP to access the discount, transition probabilities, etc.
    In particular, MDP.succAndProbReward() will be useful (see util.py for
    documentation).  Note that |V| is a dictionary.
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    Q, gamma = 0.0, mdp.discount()
    for newState, prob, reward in mdp.succAndProbReward(state,action):
        Q += prob * (reward + gamma * V[newState])
    return Q
    #raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 3.1.2

def policyEvaluation(mdp, V, pi, epsilon=0.001):
    """
    Return the value of the policy |pi| up to error tolerance |epsilon|.
    Initialize the computation with |V|.  Note that |V| and |pi| are
    dictionaries.
    """
    # BEGIN_YOUR_CODE (around 8 lines of code expected)
    Value, prevValue, gamma, convergence = {}, V.copy(), mdp.discount(), False
    while not convergence:
        convergence = True
        for state in mdp.states:
            Value[state] = 0.0
            for newState, prob, reward in mdp.succAndProbReward(state,pi[state]):
                Value[state] +=  prob * (reward + gamma * prevValue[newState])
            if float(abs(Value[state] - prevValue[state])) >= epsilon: convergence = False
        prevValue = Value.copy()
    return Value
    #raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 3.1.3

def computeOptimalPolicy(mdp, V):
    """
    Return the optimal policy based on V(state).
    You might find it handy to call computeQ().  Note that |V| is a
    dictionary.
    """
    # BEGIN_YOUR_CODE (around 4 lines of code expected)
    pi = {}
    for state in mdp.states:
        actionValue = [(action, computeQ(mdp,V,state,action)) for action in mdp.actions(state)]
        pi[state] = max(actionValue, key = itemgetter(1))[0]
    return pi
    #raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 3.1.4

class PolicyIteration(util.MDPAlgorithm):
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        # compute |V| and |pi|, which should both be dicts
        # BEGIN_YOUR_CODE (around 8 lines of code expected)
        V, pi = dict(zip(mdp.states, [0.0]*len(mdp.states))), {}
        convergence = False
        pi = computeOptimalPolicy(mdp, V)
        while not convergence:
            V = policyEvaluation(mdp, V, pi, epsilon)
            pi_new = computeOptimalPolicy(mdp, V)
            if pi_new == pi: convergence = True
            pi = pi_new
        #raise Exception("Not implemented yet")
        # END_YOUR_CODE
        self.pi = pi
        self.V = V

############################################################
# Problem 3.1.5

class ValueIteration(util.MDPAlgorithm):
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        # BEGIN_YOUR_CODE (around 11lines of code expected)
        V, pi = dict(zip(mdp.states, [0.0]*len(mdp.states))), {}
        prevV, convergence, gamma = V.copy(), False, mdp.discount()
        pi = computeOptimalPolicy(mdp, V)
        while not convergence:
            convergence = True
            for state in mdp.states:
                V[state] = 0.0
                for newState, prob, reward in mdp.succAndProbReward(state,pi[state]):
                    V[state] +=  prob * (reward + gamma * prevV[newState])
                if float(abs(V[state] - prevV[state])) >= epsilon: convergence = False
            pi = computeOptimalPolicy(mdp, V)
            prevV = V.copy()
        #raise Exception("Not implemented yet")
        # END_YOUR_CODE
        self.pi = pi
        self.V = V

############################################################
# Problem 3.1.6

# If you decide 1f is true, prove it in writeup.pdf and put "return None" for
# the code blocks below.  If you decide that 1f is false, construct a
# counterexample by filling out this class and returning an alpha value in
# counterexampleAlpha().
class CounterexampleMDP(util.MDP):
    def __init__(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        return
        #raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def startState(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        return 0
        #raise Exception("Not implemented yet")
        # END_YOUR_CODE

    # Return set of actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        return ['stay', 'quit']
        #raise Exception("Not implemented yet")
        # END_YOUR_CODE

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        if(state == 1):
            return []
        if(action == 'stay'):
            return [(state, 1, 0)]
        else: #quit
           return [(1, 0.16, 10) , (0, 0.84, 0)]
        #raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def discount(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        return 0.8
        #raise Exception("Not implemented yet")
        # END_YOUR_CODE

def counterexampleAlpha():
    # BEGIN_YOUR_CODE (around 1 line of code expected)
    return 0.1
    #raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 3.2.1

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.  The second element is the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.  The final element
    # is the current deck.
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to (0,).
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (around 40 lines of code expected)
        def quit(stateSum, prob):
            terminalState = (stateSum, None, (0,))
            if stateSum > self.threshold: return [(terminalState, prob, 0)]
            else: return [(terminalState, prob, stateSum)]
        cardsRemaining = sum(state[2])
        if cardsRemaining == 0: return []
        transition = []
        if action == 'Take':
            if state[1] != None:
                newStateSum = state[0] + self.cardValues[state[1]]
                if newStateSum > self.threshold: return quit(newStateSum, 1)
                newStateDeck = [ele for ele in state[2]]
                newStateDeck[state[1]] -= 1
                newState = (newStateSum, None, tuple(newStateDeck))
                return [(newState, 1, 0)]
            for i in range(len(self.cardValues)):
                if state[2][i] != 0:
                    prob = float(state[2][i])/cardsRemaining if cardsRemaining > 1 else 1
                    newStateSum = state[0] + self.cardValues[i]
                    if newStateSum > self.threshold:
                        transition.append(((newStateSum, None, (0,)), prob, 0))
                    else:
                        newStateDeck = [ele for ele in state[2]]
                        newStateDeck[i] -= 1
                        newState = (newStateSum, None, tuple(newStateDeck))
                        if cardsRemaining == 1: transition.append((newState, prob, newStateSum))
                        else: transition.append((newState, prob, 0))
            return transition
        elif action == 'Peek':
            if state[1] != None: return [] # already peeked once before
            for i in range(len(self.cardValues)):
                if state[2][i] != 0:
                    newState = (state[0], i, state[2])
                    prob = float(state[2][i])/cardsRemaining if cardsRemaining > 1 else 1
                    transition.append((newState, prob, -self.peekCost))
            return transition
        elif action == 'Quit': return quit(state[0], 1)
        #raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Problem 3.2.2

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the optimal action at
    least 10% of the time.
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    return BlackjackMDP(cardValues=[21,1,2,3,4], multiplicity=2,threshold=20, peekCost=1)
    #raise Exception("Not implemented yet")
    # END_YOUR_CODE
