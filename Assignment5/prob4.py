from bayesian.bbn import build_bbn

def fAge(A):
    '''Age'''
    if A == 'G1':
        return 0.45
    else:
        return 0.55


def fRace(R):
    '''Race'''
    if R == 'W':
        return 0.6
    elif(R == 'B'):
        return 0.2
    elif(R == 'A'):
        return 0.08
    else:
        return 0.12

def fGen(G):
    '''gender'''
    if G == 'M':
        return 0.51
    else:
        return 0.49

def fEduc(E):
    if(E == 'college'):
        return 0.4
    else:
        return 0.6


def fO(A, R, O):
    '''org/ age, race'''
    table = dict()
    table['tttt'] = 0.2
    table['ttft'] = 0.9
    table['tftt'] = 0.2
    table['tfft'] = 0.2
    table['fttt'] = 0.3
    table['ftft'] = 0.7
    table['fftt'] = 0.3
    table['ffft'] = 0.3
    table['tttf'] = 0.8
    table['ttff'] = 0.1
    table['tftf'] = 0.8
    table['tfff'] = 0.8
    table['fttf'] = 0.7
    table['ftff'] = 0.3
    table['fftf'] = 0.7
    table['ffff'] = 0.7

    key = ''
    key = key + 't' if A == 'G1' else key + 'f'
    key = key + 't' if R ==  'W' or R == 'B' else key + 'f'
    key = key + 't' if R ==  'W' or R == 'A' else key + 'f'
    key = key + 't' if O == 'NAACP' else key + 'f'
    return table[key]


def fK(E, K):
    '''school/educ'''
    table = dict()
    table['tt'] = 0.3
    table['tf'] = 0.6
    table['ft'] = 0.7
    table['ff'] = 0.4
    key = ''
    key = key + 't' if E == 'college' else key + 'f'
    key = key + 't' if K == 'public' else key + 'f'
    return table[key]


def fL(A, E, L):
    '''loc/age,educ'''
    table = dict()
    table['ttt'] = 0.9
    table['ttf'] = 0.6
    table['tft'] = 0.7
    table['tff'] = 0.4
    table['ftt'] = 0.1
    table['ftf'] = 0.4
    table['fft'] = 0.3
    table['fff'] = 0.6
    key = ''
    key = key + 't' if A == 'G1' else key + 'f'
    key = key + 't' if E == 'college' else key + 'f'
    key = key + 't' if L == 'urban' else key + 'f'
    return table[key]



def fW(R, W):
    '''Welfare/race'''
    table = dict()
    table['ttt'] = 0.1
    table['ttf'] = 0.3
    table['tft'] = 0.1
    table['tff'] = 0.2
    table['ftt'] = 0.9
    table['ftf'] = 0.7
    table['fft'] = 0.9
    table['fff'] = 0.8
    key = ''
    key = key + 't' if W == 'yes' else key + 'f'
    key = key + 't' if R ==  'W' or R == 'B' else key + 'f'
    key = key + 't' if R ==  'W' or R == 'A' else key + 'f'
    return table[key]


def fI(G, E, I):
    '''income/gender,educ'''
    table = dict()
    table['ttt'] = 0.2
    table['ttf'] = 0.5
    table['tft'] = 0.4
    table['tff'] = 0.6
    table['ftt'] = 0.8
    table['ftf'] = 0.5
    table['fft'] = 0.6
    table['fff'] = 0.4
    key = ''
    key = key + 't' if G == 'M' else key + 'f'
    key = key + 't' if E == 'college' else key + 'f'
    key = key + 't' if I == 'G1' else key + 'f'
    return table[key]


def fLb(K, O, W, Lb):
    '''liberal/k,o,w'''
    table = dict()
    table['tttt'] = 0.9
    table['ttft'] = 0.6
    table['tftt'] = 0.7
    table['tfft'] = 0.5
    table['fttt'] = 0.5
    table['ftft'] = 0.3
    table['fftt'] = 0.5
    table['ffft'] = 0.1
    table['tttf'] = 0.1
    table['ttff'] = 0.4
    table['tftf'] = 0.3
    table['tfff'] = 0.5
    table['fttf'] = 0.5
    table['ftff'] = 0.7
    table['fftf'] = 0.5
    table['ffff'] = 0.9

    key = ''
    key = key + 't' if K == 'public' else key + 'f'
    key = key + 't' if O ==  'NAACP'  else key + 'f'
    key = key + 't' if W ==  'yes'  else key + 'f'
    key = key + 't' if Lb == 'yes' else key + 'f'
    return table[key]



def fCs(L, I, Cs):
    '''Cs/loc,income'''
    table = dict()
    table['ttt'] = 0.3
    table['ttf'] = 0.9
    table['tft'] = 0.1
    table['tff'] = 0.4
    table['ftt'] = 0.7
    table['ftf'] = 0.1
    table['fft'] = 0.9
    table['fff'] = 0.6
    key = ''
    key = key + 't' if L == 'urban' else key + 'f'
    key = key + 't' if I == 'G1' else key + 'f'
    key = key + 't' if Cs == 'yes' else key + 'f'
    return table[key]


def fV(Lb, Cs, V):
    '''liberal/k,o,w'''
    table = dict()
    table['tttt'] = 0.4
    table['ttft'] = 0.4
    table['tftt'] = 0.7
    table['tfft'] = 0.2
    table['fttt'] = 0.2
    table['ftft'] = 0.7
    table['fftt'] = 0.4
    table['ffft'] = 0.4
    table['tttf'] = 0.2
    table['ttff'] = 0.2
    table['tftf'] = 0.1
    table['tfff'] = 0.1
    table['fttf'] = 0.1
    table['ftff'] = 0.1
    table['fftf'] = 0.2
    #table['ffff'] = 0.2


    key = ''
    key = key + 't' if Lb == 'yes' else key + 'f'
    key = key + 't' if Cs ==  'yes'  else key + 'f'
    key = key + 't' if V ==  'D' or V == 'T' else key + 'f'
    key = key + 't' if V == 'D' or V == 'R'else key + 'f'
    return table[key]




if __name__ == '__main__':
    g = build_bbn(
        fAge, fRace, fGen, fEduc, fO, fK, fL, fW, fI, fLb, fCs,fV,
        domains={
            'A': ['G1', 'G2'] ,
            'R':['W', 'B', 'A', 'H'],
            'G': ['M', 'F'],
            'E': ['college','nocollege'],
            'O': ['NAACP', 'AARP'],
            'K': ['Public', 'Private'],
            'L': ['urban', 'suburb'],
            'W': ['yes', 'no'],
            'Lb': ['yes', 'no'],
            'Cs': ['yes', 'no'],
            'I': ['G1','G2'],
            'V':['R','D','T']})
    g.q()
    g.q(R = 'A')
    g.q(O = 'NAACP')
    g.q(L = 'urban')
