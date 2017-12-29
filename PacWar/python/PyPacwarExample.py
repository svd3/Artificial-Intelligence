import _PyPacwar
import numpy


# Example Python module in C for Pacwar
def main():

	ones   = [1]*50
	threes = [3]*50
	news = "11111111111111111111111111111111111111111111000111"
	mygene = [eval(s) for s in news]
	print "Example Python module in C for Pacwar"
	print "all ones versus all threes ..."
	(rounds,c1,c2) = _PyPacwar.battle(ones, mygene)
	print "Number of rounds:", rounds 
	print "Ones PAC-mites remaining:", c1
	print "Threes PAC-mites remaining:", c2



if __name__ == "__main__": main()

