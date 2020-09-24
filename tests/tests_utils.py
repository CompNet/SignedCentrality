'''
Created on Sep 23, 2020

@author: nejat
'''

import util
import numpy as np

if __name__ == '__main__':
    print(util.which([False, True, True, False, True]))
    
    a = [1,2,4,1,2]
    print(np.isin(a, 2))
    print(util.which(np.isin(a, 2)))
    
    membership = (1,2,1,2,1,4,3,3,4,3,2,1,1,1,2,3)
    n = len(membership)
    l0 = max(membership)
    print(util.compute_prop_neg(n, l0, membership))