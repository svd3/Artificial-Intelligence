from GeneticAlgo import *
import time

resID = int(time.time())

t0 = time.time()
benchmark = readBenchmark("benchmark.txt")

randBench = randomInit(1000)
for gene in randBench:
	benchmark.append(gene)

newPopulation,fit = runGeneticAlgo(20,10,benchmark)
writeResult("Results/result"+str(resID)+".txt", newPopulation, fit)

t1 = time.time()
print t1 - t0, "sec"
