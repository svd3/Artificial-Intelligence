import re, util

class Board:
	def __init__(self, description):
		lines = description.split("\n")
		self.numLizards = lines[1]
		self.gridSize = lines[0]
		self.grid = []
		flag = True
		for line in description.split("\n"):
			if flag:
				self.numLizards = eval(line)
				flag = False
			else:
				r = len(self.grid)
				row = []
				for cell in line.split(" "):
					row.append(eval(cell))
				self.grid.append(row)
		self.gridSize = len(self.grid)



class Lizard(util.SearchProblem):
	# |scenario|: board specification.
	def __init__(self, scenario):
	self.scenario = scenario
		
	# Return the start state.
	def startState(self):
		state = ()
		return state
	
	# Return whether |state| is a goal state or not.
	def isGoal(self, state):
		return len(state) == self.scenario.numLizards

	# Return a list of (action, newState, cost) tuples corresponding to edges
	# coming out of |state|.
	def succAndCost(self, state):
		
		
