from GeneticAlgo import *
import sys
import numpy as np
# Code Testing
benchmark = readBenchmark("benchmark.txt")
#print "Gene: "
#str1 = sys.stdin.readline()
#mygene = [eval(s) for s in str1 if s!='\n']
#for gene in benchmark:
#	print matchScore(mygene,gene)


#print Z

def writeResult(filename, Z):
	f = open(filename, 'w')
	N = len(Z)
	str1 = ''.join(str(Z[i][j]) if j!=(N-1) else str(Z[i][j]) + '\n' for i in range(N) for j in range(N))
	f.write(str1)
	f.close()
	return

Z = (np.zeros((4,4), dtype=np.int8))
writeResult("out123.txt", Z)
