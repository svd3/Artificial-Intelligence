import numpy as np
import scipy.spatial as ss

       
def energy(protein, actions):
    locs = np.zeros([len(protein), 2])
    temp = [0,0]
    for i in range(len(protein)-1):
        if( actions[i] == 2):
            temp = np.add(temp, [0,1])
        elif(actions[i] == -2):
            temp = np.add(temp, [0,-1])
        elif(actions[i] == 1):
            temp = np.add(temp, [1,0]) 
        else:
            temp = np.add(temp, [-1,0])
        locs[i+1,:] = temp
    Hlocs = locs[np.where(np.array(protein) == 1),:]
    dists = ss.distance.pdist(Hlocs[0])
    return sum(dists)
    
    
def checker(protein, actions):
    locs = np.zeros([len(protein), 2])
    temp = [0,0]
    for i in range(len(protein)-1):
        if( actions[i] == 2):
            temp = np.add(temp, [0,1])
        elif(actions[i] == -2):
            temp = np.add(temp, [0,-1])
        elif(actions[i] == 1):
            temp = np.add(temp, [1,0]) 
        else:
            temp = np.add(temp, [-1,0])
        locs[i+1,:] = temp
    ulen = len(np.array(list(set(tuple(p) for p in locs))))
    return (ulen == len(locs))
    
    
#Main program            
protein = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0]
#protein = [0,1,1,1,1,0]
conf = [[1]] #initialization for possible conformations
actions = [1,-1,2,-2]
U = 0
Z = 0
p1 = 0.7
p2 = 0.5



for l in range(3,len(protein)+1):#len(protein)):
    print(l)
    temp = conf
    conf = []
    energies = []
    for c in temp:
        poss_actions = actions[:]
        poss_actions.remove(-c[-1]) #valid actions, have to add interference
        for acts in poss_actions:
            if(checker(protein[:l], np.append(c,acts))):
                conf.append( np.append(c,acts) )
                energies.append(energy(protein[:l], np.append(c,acts)))
    #print(conf)
    print(len(energies))
    temp = []
    U = np.min(energies)
    Z = np.mean(energies)    

    if(protein[l-1] == 1): #Hydrophobic
        for e in range(len(energies)):
            egy = energies[e]
            if(egy > Z):
                r = np.random.uniform(0,1)
                if(r > p1):
                    temp.append(conf[e])
            elif(egy < U):
                temp.append(conf[e])
            else:
                r = np.random.uniform(0,1)
                if(r > p2):
                    temp.append(conf[e])
        conf = temp   
        

print(min(energies))
