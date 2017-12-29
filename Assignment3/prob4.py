import collections, util, math, random
from submission import computeOptimalPolicy
from operator import itemgetter
# Problem 4
""" Problem 4 """

class RobotMDP(util.MDP):
    def __init__(self, discount, Prob, Reward):
        self.discountFactor = discount
        self.Prob = Prob
        self.Reward = Reward

    def startState(self): return 1

    def actions(self, state):
        if state != 2: return ['C','R','E']
        else: return ['C','R']

    def succAndProbReward(self, state, action):
        transition =[]
        a = {'C':0, 'R':1, 'E':2}[action]
        for newState in [1,2,3]:
            transition.append((newState, self.Prob[state-1][a][newState-1],
                                        self.Reward[state-1][a][newState-1]))
        return transition

    def discount(self): return self.discountFactor


class myValueIteration(util.MDPAlgorithm):
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        V, pi = dict(zip(mdp.states, [0.0]*len(mdp.states))), {}
        prevV = V.copy()
        convergence = False
        gamma = mdp.discount()
        pi = computeOptimalPolicy(mdp, V)
        t = 0
        print "time",t, ":\tV:", V
        while not convergence:
            for state in mdp.states:
                V[state] = 0.0
                for newState, prob, reward in mdp.succAndProbReward(state, pi[state]):
                    V[state] +=  prob * (reward + gamma * prevV[newState])
            # Values updated
            t += 1
            print "time",t, ":\tV:", V
            pi_new = computeOptimalPolicy(mdp, V)
            if pi_new == pi: convergence = True
            pi = pi_new
            prevV = V.copy()
        self.pi = pi
        self.V = V

if __name__ == "__main__":
    # Getting input
    f = open("input.txt",'r')
    lines = f.readlines()
    f.close()
    k = 1 # skip first line
    Prob = []
    for i in range(3):
        actionProb = []
        for a in range(3):
            if i!=1 or a != 2:
                str1 = str.split(lines[k])
                actionProb.append([eval(s) for s in str1])
                k += 1
        Prob.append(actionProb)
    k += 1 # skip one more line
    Reward = []
    for i in range(3):
        actionReward = []
        for a in range(3):
            if i!=1 or a != 2:
                str1 = str.split(lines[k])
                actionReward.append([eval(s) for s in str1])
                k+=1
        Reward.append(actionReward)
    discounts = [1, 0.75, 0.5, 0.1]
    for discount in discounts:
        print "Discount :", discount
        mdp = RobotMDP(discount, Prob, Reward)
        vi = myValueIteration()
        vi.solve(mdp)
        print "pi :", vi.pi
