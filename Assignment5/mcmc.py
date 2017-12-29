import numpy as np
from collections import namedtuple
Q = np.array([[17./27, 5./54, 5./18, 0.],
		      [11./27, 22./189, 0., 10./21],
              [2./9, 0., 59./153, 20./51],
              [0., 1./42, 11./102, 310./357]])
Q0 = Q

states = ((True, True), (True, False), (False, True), (False, False))
samples = []
index = np.random.choice(range(len(states))) #random start state
samples.append(states[index])
N = int(raw_input("Enter N: "))
count = 0
for i in range(N):
	index = np.random.choice(range(len(states)), p = Q[index])
	state = states[index]
	samples.append(state)
	if i >= 50:
		#burn-in period
		if state[1]: count += 1

p = float(count)/(N - 50)
print "probability estimates"
print "P(r | s, w) =", p
print "P(_r | s, w) =", 1.0 - p
