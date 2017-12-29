
f = open("input.txt",'r')
lines = f.readlines()
f.close()
k = 0
Prob = []
for i in range(3):
    actionProb = []
    for a in range(3):
        if i!=1 or a != 2:
            str1 = str.split(lines[k])
            actionProb.append([eval(s) for s in str1])
            k+=1
    Prob.append(actionProb)

Reward = []
for i in range(3):
    actionReward = []
    for a in range(3):
        if i!=1 or a != 2:
            str1 = str.split(lines[k])
            actionReward.append([eval(s) for s in str1])
            k+=1
    Reward.append(actionReward)


print Prob[0][0]

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
        if cardsRemaining == 1:
            newStateSum = state[0] + self.cardValues[state[2].index(1)]
            if newStateSum > self.threshold: return quit(newStateSum, 1)
            newStateDeck = [ele for ele in state[2]]
            newStateDeck[state[2].index(1)] -= 1
            newState = (newStateSum, None, tuple(newStateDeck))
            return [(newState, 1, newStateSum)]
        if state[1] != None:
            newStateSum = state[0] + self.cardValues[state[1]]
            if newStateSum > self.threshold: return quit(newStateSum, 1)
            newStateDeck = [ele for ele in state[2]]
            newStateDeck[self.cardValues.index(state[1])] -= 1
            newState = (newStateSum, None, newStateDeck)
            return [(newState, 1, 0)]
        for i in range(len(self.cardValues)):
            newStateSum, newStateDeck = state[0], [ele for ele in state[2]]
            if state[2][i] != 0:
                prob = float(state[2][i])/cardsRemaining if cardsRemaining > 1 else 1
                newStateSum += self.cardValues[i]
                reward = 0
                if newStateSum > self.threshold: newStateDeck = (0,)
                else:
                    if cardsRemaining == 1: reward = newStateSum
                    newStateDeck[i] -= 1
                newState = (newStateSum, None, tuple(newStateDeck))
                transition.append((newState, prob, reward))
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
