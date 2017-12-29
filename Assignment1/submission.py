import re, util

############################################################
# Problem 1a: UCS test case

# Return an instance of util.SearchProblem.
# You might find it convenient to use
# util.createSearchProblemFromString.
def createUCSTestCase(n):
    # BEGIN_YOUR_CODE (around 5 lines of code expected)
    s = ""
    for i in range(n):
        s = s + "n0 n" + str(i+1) + " 1\n"
    s = s + "n" + str(i+1) + " n" + str(i+2) + " 2\n"
    return util.createSearchProblemFromString("n0", "n2", s)
    # END_YOUR_CODE

############################################################
# Problem 1b: A-star search

# Takes the SearchProblem |problem| you're trying to solve and a |heuristic|
# (which is a function that maps a state to an estimate of the cost to the
# goal).  Returns another search problem |newProblem| such that running uniform
# cost search on |newProblem| is equivalent to running A* on |problem| with
# |heuristic|.
def astarReduction(problem, heuristic):
    class NewSearchProblem(util.SearchProblem):
        # Please refer to util.SearchProblem to see the functions you need to
        # override.
        # BEGIN_YOUR_CODE (around 9 lines of code expected)
        def startState(self): return problem.startState()
        def isGoal(self, state): return problem.isGoal(state)
        def succAndCost(self, state):
            graph = problem.succAndCost(state)
            graph = [(node[0], node[1], node[2] + heuristic(node[1]) - heuristic(state)) for node in graph]
            return graph
        # END_YOUR_CODE
    newProblem = NewSearchProblem()
    return newProblem

# Implements A-star search by doing a reduction.
class AStarSearch(util.SearchAlgorithm):
    def __init__(self, heuristic):
        self.heuristic = heuristic

    def solve(self, problem):
        # Reduce the |problem| to |newProblem|, which is solved by UCS.
        newProblem = astarReduction(problem, self.heuristic)
        algorithm = util.UniformCostSearch()
        algorithm.solve(newProblem)

        # Copy solution back
        self.actions = algorithm.actions
        if algorithm.totalCost != None:
            self.totalCost = algorithm.totalCost + self.heuristic(problem.startState())
        else:
            self.totalCost = None
        self.numStatesExplored = algorithm.numStatesExplored

############################################################
# Problem 2b: Delivery

class DeliveryProblem(util.SearchProblem):
    # |scenario|: delivery specification.
    def __init__(self, scenario):
        self.scenario = scenario

    # Return the start state.
    def startState(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        return (self.scenario.truckLocation, tuple([0]*self.scenario.numPackages))
        # END_YOUR_CODE

    # Return whether |state| is a goal state or not.
    def isGoal(self, state):
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        goal = (self.scenario.truckLocation, tuple([2]*self.scenario.numPackages))
        return state == goal
        # END_YOUR_CODE

    # Return a list of (action, newState, cost) tuples corresponding to edges
    # coming out of |state|.
    def succAndCost(self, state):
        # Hint: Call self.scenario.getNeighbors((x,y)) to get the valid neighbors
        # at that location. In order for the simulation code to work, please use
        # the exact strings 'Pickup' and 'Dropoff' for those two actions.
        # BEGIN_YOUR_CODE (around 18 lines of code expected)
        pickupLocations = self.scenario.pickupLocations
        dropoffLocations = self.scenario.dropoffLocations
        packagesPicked = state[1].count(1)
        graph = self.scenario.getNeighbors(state[0])
        graph = [(node[0], (node[1], state[1]), 1 + packagesPicked) for node in graph]
        if state[0] in pickupLocations:
            i = pickupLocations.index(state[0])
            if state[1][i] == 0:
                pickupState = (state[0], state[1][:i] + (1,) + state[1][i+1:])
                graph.append(('Pickup', pickupState, 0))
        if state[0] in dropoffLocations:
            i = dropoffLocations.index(state[0])
            if state[1][i] == 1:
                dropoffState = (state[0], state[1][:i] + (2,) + state[1][i+1:])
                graph.append(('Dropoff', dropoffState, 0))
        return graph
        # END_YOUR_CODE

############################################################
# Problem 2c: heuristic 1


# Return a heuristic corresponding to solving a relaxed problem
# where you can ignore all barriers and not do any deliveries,
# you just need to go home
def createHeuristic1(scenario):
    def heuristic(state):
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        start = scenario.truckLocation
        return abs(state[0][0] - start[0]) + abs(state[0][1] - start[1])
        # END_YOUR_CODE
    return heuristic

############################################################
# Problem 2d: heuristic 2

# Return a heuristic corresponding to solving a relaxed problem
# where you can ignore all barriers, but
# you'll need to deliver the given |package|, and then go home
def createHeuristic2(scenario, package):
    def heuristic(state):
        # BEGIN_YOUR_CODE (around 11 lines of code expected)
        start = scenario.truckLocation
        pickup = scenario.pickupLocations[package]
        drop = scenario.dropoffLocations[package]
        if state[1][package] == 0:
            h = abs(state[0][0] - pickup[0]) + abs(state[0][1] - pickup[1])
            h += 2*abs(pickup[0] - drop[0]) + 2*abs(pickup[1] - drop[1])
            h += abs(drop[0] - start[0]) + abs(drop[1] - start[1])
            return h
        if state[1][package] == 1:
            h = 2*abs(state[0][0] - drop[0]) + 2*abs(state[0][1] - drop[1])
            h += abs(drop[0] - start[0]) + abs(drop[1] - start[1])
            return h
        if state[1][package] == 2:
            h = abs(state[0][0] - start[0]) + abs(state[0][1] - start[1])
            return h
        # END_YOUR_CODE
    return heuristic

############################################################
# Problem 2e: heuristic 3

# Return a heuristic corresponding to solving a relaxed problem
# where you will delivery the worst(i.e. most costly) |package|,
# you can ignore all barriers.
# Hint: you might find it useful to call
# createHeuristic2.
def createHeuristic3(scenario):
    # BEGIN_YOUR_CODE (around 5 lines of code expected)
    def heuristic(state):
        max = 0
        for package in range(scenario.numPackages):
            h = createHeuristic2(scenario,package)
            if h(state) > max: max = h(state)
        return max
    return heuristic
    # END_YOUR_CODE
