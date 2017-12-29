import _PyPacwar
import numpy as np
import time
#from numpy import random

# Scoring a match between two genes
def matchScore(geneA, geneB):
	(rounds,c1,c2) = _PyPacwar.battle(geneA, geneB)
	
	if c2 == 0:
		# A wins, score based on rounds
		if rounds < 100: scoreA = 20
		elif rounds < 200: scoreA = 19
		elif rounds < 300: scoreA = 18
		else: scoreA = 17
	elif c1 == 0:
		# B wins, score based on rounds
		if rounds < 100: scoreA = 0
		elif rounds < 200: scoreA = 1
		elif rounds < 300: scoreA = 2
		else: scoreA = 3
	else:
		if c1 >= c2:
			r = c1/c2
			if r >= 10: scoreA = 13
			elif r >= 3: scoreA = 12
			elif r>= 1.5: scoreA = 11
			else: scoreA = 10
		else:
			r = c1/c2
			if r >= 10: scoreA = 7
			elif r >= 3: scoreA = 8
			elif r >= 1.5: scoreA = 9
			else: scoreA = 10
	return (scoreA, 20 - scoreA)

# Initialize a random gene sequence
def randomInit(N):
	return np.random.randint(0,4,50*N).reshape(N,50).tolist()

# Evaluate the population against some benchmark cases
def evalPopulation(population, benchmark):
	N = len(population)
	K = len(benchmark)
	fitness = [10.0]*N
	for i in range(N):
		for j in range(K):
			scorePopI, scoreBenchJ = matchScore(population[i],benchmark[j])
			fitness[i] += scorePopI
		fitness[i] = float(fitness[i])*100.0/(20.0*K) # Normalizing to 100 scale
	return fitness


# Evaluate the population by holding matches between all pairs and adding up scores
def evalPopulation_within(population):
	N = len(population)
	fitness = [10.0]*N
	for i in range(N):
		for j in range(i+1,N):
			scoreI, scoreJ = matchScore(population[i],population[j])
			fitness[j] += scoreJ
			fitness[i] += scoreI
	return fitness

# Select parents for crossover from the current population based on their fitness
# The probabilities of selection of an individual gene is proportional to fitness (for now)
def selection(population, fitness):
	N = len(population)
	S = sum(fitness)
	prob = [float(val)/S for val in fitness]
	choice = np.random.choice(N, N, p = prob)
	np.random.shuffle(choice)	# not necessary to shuffle, they're already randomly placed
	newPopulation = [population[i] for i in choice]
	return newPopulation

# Crossover parent genes to make children. We're doing uniform crossover
def crossover(population):
	N = len(population)
	# Uniform Crossover
	probSwap = 0.75
	for i in range(N/2):
		for j in range(50):
			if np.random.uniform() < probSwap:
				temp = population[i][j]
				population[i][j] = population[N-1-i][j]
				population[N-1-i][j] = temp
	return

# Mutate the current population. Only 1 bit in 50
def mutate(population):
	N = len(population)
	for i in range(N):
		k,j = np.random.randint(0,50,2)
		population[i][k] = np.random.randint(0,4)
		population[i][j] = np.random.randint(0,4)
	return

# Main code
def runGeneticAlgo(N, maxIter, benchmark):
	# Initialize population of size N
	population = randomInit(N)
	for iter in range(maxIter):
		fitness = evalPopulation(population, benchmark)
		population = selection(population, fitness) # Don't remember old population
		crossover(population)
		mutate(population)
	return population, evalPopulation(population, benchmark)

# Reading Benchmark file
def readBenchmark(filename):
	benchmark = []
	f = open(filename,'r')
	for line in f:
		benchmark.append([eval(s) for s in line if s!='\n'])
	return benchmark

# Writing Benchmark file
def writeBenchmark(filename, benchmark):
	f = open(filename, 'w')
	for gene in benchmark:
		str1 = ''.join(str(bit) for bit in gene)
		str1 = str1 + '\n'
		f.write(str1)
	f.close()
	return

def writeResult(filename, population, fitness):
	f = open(filename, 'w')
	N = len(population)
	for i in order(fitness, rev = True):
		str1 = ''.join(str(bit) for bit in population[i])
		str1 = "gene: " + str1 + '\n' + "fit: " + str(fitness[i]) + '\n'
		print str1
		f.write(str1)
	f.close()
	return

def order(a, rev = False):
	b = range(len(a))
	b = sorted(b,key = lambda x:a[x], reverse = rev)
	return b

