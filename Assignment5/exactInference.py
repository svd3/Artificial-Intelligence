'''
Licensing Information: Please do not distribute or publish solutions to this
project. You are free to use and extend Driverless Car for educational
purposes. The Driverless Car project was developed at Stanford, primarily by
Chris Piech (piech@cs.stanford.edu). It was inspired by the Pacman projects.
'''
from engine.const import Const
import util, math
import numpy as np
# Class: ExactInference
# ---------------------
# Maintain and update a belief distribution over the probability of a car
# being in a tile using exact updates (correct, but slow times).
class ExactInference(object):
    
    # Function: Init
    # --------------
    # Constructer that initializes an ExactInference object which has
    # numRows x numCols number of tiles.
    def __init__(self, numRows, numCols):
        self.skipElapse = False ### ONLY USED BY GRADER.PY in case problem 2 has not been completed
        self.belief = util.Belief(numRows, numCols)
        self.transProb = util.loadTransProb()
   
    # Function: Observe (reweight the probablities based on an observation)
    # -----------------
    # Updates beliefs based on the distance observation $d_t$ and your position $a_t$.
    # observedDist: true distance plus a mean-zero Gaussian with standard deviation Const.SONAR_STD
    # agentX: x location of your car (not the one you are tracking)
    # agentY: y location of your car (not the one you are tracking)
    #
    # Suggestion: Loop over the rows and columns and compute the true distance
    # between (row, col) and the agent.  Then use this distance to find the
    # probability from util.pdf and update the self.belief probability.
    # Note: you need to convert (row, col) into a location using util.rowToY and util.colToX.
    def observe(self, agentX, agentY, observedDist):
        # BEGIN_YOUR_CODE (around 5 lines of code expected)
        
        # END_YOUR_CODE
        for r in range(self.belief.getNumRows()):
            for c in range(self.belief.getNumCols()):
                mu = math.hypot( util.rowToY(r)-agentY , util.colToX(c)-agentX)
                p = util.pdf(mu, Const.SONAR_STD, observedDist)
                self.belief.setProb(r,c,p*self.belief.getProb(r,c))
        self.belief.normalize()

    # Function: Elapse Time (propose a new belief distribution based on a learned transition model)
    # ---------------------
    # Update your inference to handle the passing of one time step.
    # Use the transition probabilities in self.transProb.
    # belief.addProb should be helpful here.
    #
    # Suggestion: Loop over all the (oldTile, newTile) pairs (keys of
    # self.transProb), and add the probability of going to newTile
    # given the probability of being in the oldTile to newBelief.
    def elapseTime(self):
        if self.skipElapse: return ### ONLY FOR THE GRADER TO USE IN Problem 1
        newBelief = util.Belief(self.belief.getNumRows(), self.belief.getNumCols(), 0)
        # BEGIN_YOUR_CODE (around 3 lines of code expected)
        for k in self.transProb.keys():
            delta = self.transProb[k]*self.belief.getProb(k[0][0], k[0][1])
            newBelief.addProb(k[1][0], k[1][1], delta)
        # END_YOUR_CODE
        self.belief = newBelief
      
    # Function: Get Belief
    # ---------------------
    # Returns your belief of the probability that the car is in each tile. Your
    # belief probabilities should sum to 1.    
    def getBelief(self):
        return self.belief
