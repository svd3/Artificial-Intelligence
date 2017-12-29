import random, time, sys
import numpy as np
from operator import itemgetter

def getNextLocations(state):
	k = len(state)
	x,y = state[k-1]
	if k == 1: return [(x+1,y)]
	nextLocations = []
	if not (x-1,y) in state: nextLocations.append((x-1,y))
	if not (x+1,y) in state: nextLocations.append((x+1,y))
	if not (x,y-1) in state: nextLocations.append((x,y-1))
	if not (x,y+1) in state: nextLocations.append((x,y+1))
	return nextLocations

def deltaEnergy(state, newLoc, type):
	e = 0
	if type == 0: return e # hydrophillic or polar
	for i in range(len(state)):
		if chain[i] == 1:
			e += np.sqrt((newLoc[0] - state[i][0])**2 + (newLoc[1] - state[i][1])**2)
	return e

def getCenter(state):
	c = 0
	x=y=0
	for i in range(len(state)):
		if chain[i] == 1:
			x += state[i][0]
			y += state[i][1]
			c+=1
	return (x/c,y/c)

def dE2(state,loc,type):
	center = getCenter(state)
	if type==0:
		return -np.sqrt((loc[0] - center[0])**2 + (loc[1] - center[1])**2)
	else: return 0

chain1 = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0]
index = [i for i, x in enumerate(chain1) if x == 1]
chain = [chain1[i] for i in range(min(index),max(index)+1)]
print(chain)
n = len(chain)
U = [0]*n
Z = [0]*n
count = [0]*n
initState = [(0,0)]
p2 = 0.6
p1 = 0.8
def Search(state, E):
	global U,Z,count,p1,p2
	k = len(state)
	#print k
	nextLocations = getNextLocations(state)
	result =[]
	if len(nextLocations) == 0: return (False, E,state)
	else:
		for loc in nextLocations:
			dE = round(deltaEnergy(state, loc, chain[k]))
			dE3 = dE2(state,loc, chain[k])
			T1 = 1000/k
			T2 = T1/2
			p2 = p1 = np.exp(dE/T1)/(1+np.exp(dE/T1))
			p3 = np.exp(dE3/T2)/(1+np.exp(dE3/T2))
			#p3 = 0
			newE = E + dE
			newState = [x for x in state]
			newState.append(loc)
			if U[k] == 0: U[k] = newE
			else: U[k] = min(U[k], newE)
			Z[k] += newE
			count[k] += 1

			if k == (n-1): result.append((True, newE, newState)) #minE = newE
			else:
				if chain[k] == 1:
					if newE <= U[k]: result.append(Search(newState, newE))
					elif newE > (Z[k]/count[k]):
						if random.uniform(0,1) > p1: result.append(Search(newState, newE))
					else:
						if random.uniform(0,1) > p2: result.append(Search(newState, newE))
				else:
					if random.uniform(0,1) > p3: result.append(Search(newState, newE))

		temp = [ele for ele in result if ele[0]]
		if len(temp)!=0: return min(temp, key=itemgetter(1))
		else: return (False, E, state)

print n
t0 = time.time()
ans = Search(initState,0)
print ans[0]
print ans[1]
print ans[2]
t1= time.time()
print t1-t0






